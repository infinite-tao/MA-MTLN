import torch
import torch.nn as nn
import time
import os
import numpy as np
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
from datetime import datetime
import sys
import torch.backends.cudnn as cudnn

from network_architecture.MTLN import MTLN3D
from Data_preprocess import MyData, get_fold_list
from data_augmentation import default_3D_augmentation_params, default_2D_augmentation_params, get_default_augmentation, get_patch_size
from dataset_load import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset,get_case_identifiers
from Loss_functions import JI_and_Focal_loss, AutomaticWeightedLoss, softmax_helper, sum_tensor

try:
    from apex import amp
except ImportError:
    amp = None

class Trainer(nn.Module):
    def __init__(self, fold, data_root, out_path, output_fold):
        super(Trainer, self).__init__()
        self.fold = fold
        self.data_root = data_root
        self.dataset_directory = out_path
        self.output_checkpoints = self.output_folder = output_fold
        self.was_initialized = False
        self.log_file = self.use_mask_for_norm = None
        self.batch_dice = False
        self.seg_loss = JI_and_Focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False})
        self.class_loss = torch.nn.CrossEntropyLoss().cuda()
        self.aw1 = AutomaticWeightedLoss(2)
        self.fp16 = False

    def initialize(self, training=True):
        self.batch_size = 2
        self.patch_size = np.array([24, 256, 256]).astype(int)
        self.do_data_augmentation = True
        self.num_classes = 2
        self.setup_data_aug_params()

        self.initial_lr = 2e-4
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 10
        self.weight_decay = 2e-5

        self.oversample_foreground_percent = 0.33
        self.pad_all_sides = None

        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.save_every = 50
        self.save_latest_only = True
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 300
        self.num_val_batches_per_epoch = 100
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        self.val_eval_criterion_MA = None
        self.val_eval_criterion_MA_class = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_val_eval_criterion_MA_class = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.online_eval_foreground_acc = []
        self.online_eval_tp_class = []
        self.online_eval_fp_class = []
        self.online_eval_fn_class = []
        self.online_eval_tn_class = []

        if training:
            self.train_data, self.test_data = self.get_data()
            unpack_dataset(self.data_root)
            self.train_data, self.test_data = get_default_augmentation(self.train_data, self.test_data,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)
        self.network = MTLN3D().cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")
        self.was_initialized = True

    def load_dataset(folder):
        case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()
        dataset = OrderedDict()
        for c in case_identifiers:
            dataset[c] = OrderedDict()
            dataset[c]['data_file'] = os.path.join(folder, "%s.npz" % c)
            with open(os.path.join(folder, "%s.pkl" % c), 'rb') as f:
                dataset[c]['properties'] = pickle.load(f)
            if dataset[c].get('seg_from_prev_stage_file') is not None:
                dataset[c]['seg_from_prev_stage_file'] = os.path.join(folder, "%s_segs.npz" % c)
        return dataset

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = os.path.join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = test_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            test_keys = splits[self.fold]['val']

        tr_keys.sort()
        test_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_test = OrderedDict()
        for i in test_keys:
            self.dataset_test[i] = self.dataset[i]

    def load_dataset(self):
        self.dataset = load_dataset(self.data_root)

    def get_data(self):
        self.load_dataset()
        self.do_split()
        dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                             False, oversample_foreground_percent=self.oversample_foreground_percent,
                             pad_mode="constant", pad_sides=self.pad_all_sides)
        dl_test = DataLoader3D(self.dataset_test, self.patch_size, self.patch_size, self.batch_size, False,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_test

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                time.sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def setup_data_aug_params(self):
        self.data_aug_params = default_3D_augmentation_params
        if self.do_data_augmentation:
            self.data_aug_params["dummy_2D"] = True
            self.print_to_log_file("Using dummy2d data augmentation")
            self.data_aug_params["elastic_deform_alpha"] = \
                default_2D_augmentation_params["elastic_deform_alpha"]
            self.data_aug_params["elastic_deform_sigma"] = \
                default_2D_augmentation_params["elastic_deform_sigma"]
            self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        if self.do_data_augmentation:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            # print(self.basic_generator_patch_size)[301,301]
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            # print(self.basic_generator_patch_size)[24,301,301]
            patch_size_for_spatialtransform = self.patch_size[1:]
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        class_target = data_dict['class_label']

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        if not isinstance(class_target, torch.Tensor):
            class_target = torch.from_numpy(class_target).float()

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        class_target = class_target.cuda(non_blocking=True)

        self.optimizer.zero_grad()
        output = self.network(data)
        del data

        l0 = self.seg_loss(output[0], target)
        l1 = self.seg_loss(output[1], target)
        l2 = self.seg_loss(output[2], target)
        l10 = self.class_loss(output[3], class_target.squeeze().long())
        # l = self.loss(output, target)
        l3 = (0.8 * l0 + 0.9 * l1 + l2)
        l = self.aw1.cuda()(l3, l10)
        # l = l3+l10
        if run_online_evaluation:
            self.run_online_evaluation(output[2], output[3], target, class_target)

        del target

        if do_backprop:
            if not self.fp16 or amp is None:
                l.backward()
            else:
                with amp.scale_loss(l, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()

        return l.detach().cpu().numpy()
    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time.time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics)},
            fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time.time() - start_time))

    def finish_online_evaluation(self):
        ###classification
        self.online_eval_tp_class = np.sum(self.online_eval_tp_class, 0)
        self.online_eval_fp_class = np.sum(self.online_eval_fp_class, 0)
        self.online_eval_fn_class = np.sum(self.online_eval_fn_class, 0)
        self.online_eval_tn_class = np.sum(self.online_eval_tn_class, 0)

        global_acc_per_class = [i for i in [(i + h) / (i + j + k + h) for i, j, k, h in
                                            zip(self.online_eval_tp_class, self.online_eval_fp_class,
                                                self.online_eval_fn_class, self.online_eval_tn_class)]
                                if not np.isnan(i)]

        self.online_eval_foreground_acc = []
        self.online_eval_tp_class = []
        self.online_eval_fp_class = []
        self.online_eval_fn_class = []
        self.online_eval_tn_class = []
        ###segmentation
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]

        self.all_val_eval_metrics.append(np.mean(global_acc_per_class))
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Val global dc per class:", str(global_dc_per_class))
        self.print_to_log_file("Val global acc per class:", str(global_acc_per_class))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.epoch % self.save_every == (self.save_every - 1):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        if self.val_eval_criterion_MA_class is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA_class = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA_class = self.all_val_eval_metrics[-2]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower 
                is better, so we need to negate it. 
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
                self.val_eval_criterion_MA_class = self.val_eval_criterion_alpha * self.val_eval_criterion_MA_class - (
                        1 - self.val_eval_criterion_alpha) * \
                                                   self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]
                self.val_eval_criterion_MA_class = self.val_eval_criterion_alpha * self.val_eval_criterion_MA_class + (
                        1 - self.val_eval_criterion_alpha) * \
                                                   self.all_val_eval_metrics[-2]
    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.best_val_eval_criterion_MA_class = self.val_eval_criterion_MA_class

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)
            self.print_to_log_file(
                "current best_val_eval_criterion_MA_class is %.4f0" % self.best_val_eval_criterion_MA_class)
            self.print_to_log_file("current val_eval_criterion_MA_class is %.4f" % self.val_eval_criterion_MA_class)

            if self.val_eval_criterion_MA_class > self.best_val_eval_criterion_MA_class:
                self.best_val_eval_criterion_MA_class = self.val_eval_criterion_MA_class
                self.print_to_log_file("saving best classification epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best_class.model"))

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best segmentation epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best_seg.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                self.print_to_log_file(
                    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def on_epoch_end(self):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        # self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def run_online_evaluation(self, output, output_class, target, target_class):
        with torch.no_grad():
            ###classification
            num_classes0 = output_class.shape[1]
            output_class_softmax = softmax_helper(output_class)
            output_class = output_class_softmax.argmax(1)
            target_class = target_class[:, 0]
            axes_class = tuple(range(1, len(target_class.shape)))
            tp_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            fp_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            fn_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            tn_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            for c in range(1, num_classes0):
                tp_hard_class[:, c - 1] = sum_tensor((output_class == c).float() * (target_class == c).float(),
                                                     axes=axes_class)
                fp_hard_class[:, c - 1] = sum_tensor((output_class == c).float() * (target_class != c).float(),
                                                     axes=axes_class)
                fn_hard_class[:, c - 1] = sum_tensor((output_class != c).float() * (target_class == c).float(),
                                                     axes=axes_class)
                tn_hard_class[:, c - 1] = sum_tensor((output_class != c).float() * (target_class != c).float(),
                                                     axes=axes_class)

            tp_hard_class = tp_hard_class.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard_class = fp_hard_class.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard_class = fn_hard_class.sum(0, keepdim=False).detach().cpu().numpy()
            tn_hard_class = tn_hard_class.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_acc.append(list((tp_hard_class + tn_hard_class) / (
                    tp_hard_class + fp_hard_class + fn_hard_class + tn_hard_class + 1e-8)))
            self.online_eval_tp_class.append(list(tp_hard_class))
            self.online_eval_fp_class.append(list(fp_hard_class))
            self.online_eval_fn_class.append(list(fn_hard_class))
            self.online_eval_tn_class.append(list(tn_hard_class))

            ###segmentation
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def run_trainer(self):
        torch.cuda.empty_cache()
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if not self.was_initialized:
            self.initialize(True)
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time.time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            for b in range(self.num_batches_per_epoch):
                l = self.run_iteration(self.train_data, do_backprop=True, run_online_evaluation=False)
                train_losses_epoch.append(l)
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.test_data, do_backprop=False,
                                           run_online_evaluation=True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("val loss (train=False): %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("val loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            epoch_end_time = time.time()

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()
            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            if self.epoch % 5 == 0:
                self.save_checkpoint(join(self.output_folder, 'epoch_{}_model_best.model'.format(self.epoch)))
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))


