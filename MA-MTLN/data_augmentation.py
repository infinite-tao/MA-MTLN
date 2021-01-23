from copy import deepcopy
import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose, RenameTransform, GammaTransform, SpatialTransform
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms import MirrorTransform, NumpyToTensor
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
# from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
#     RemoveRandomConnectedComponentFromOneHotEncodingTransform, ApplyRandomBinaryOperatorTransform
# from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
#     MaskTransform


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "do_rotation": True,
    "rotation_x": (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "num_threads": 12,
    "num_cached_per_thread": 1,
    "mirror": True,
    "mirror_axes": (0, 1, 2),
    "p_eldef": 0.2,
    "p_scale": 0.2,
    "p_rot": 0.2,
    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "all_segmentation_labels": None,  # used for pyramid
    "move_last_seg_chanel_to_data": False,  # used for pyramid
    "border_mode_data": "constant",
    "advanced_pyramid_augmentations": False  # used for pyramid
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180./360 * 2. * np.pi, 180./360 * 2. * np.pi)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (0, 1) # this can be (0, 1, 2) if dummy_2D=True


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90/360 * 2. * np.pi, rot_x)
    rot_y = min(90/360 * 2. * np.pi, rot_y)
    rot_z = min(90/360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((rotate_coords_3d(coords, rot_x, 0, 0), final_shape)), 0)
        final_shape = np.max(np.vstack((rotate_coords_3d(coords, 0, rot_y, 0), final_shape)), 0)
        final_shape = np.max(np.vstack((rotate_coords_3d(coords, 0, 0, rot_z), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((rotate_coords_2d(coords, rot_x), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def get_default_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params, border_val_seg=-1, pin_memory=True,
                             seeds_train=None, seeds_val=None):
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot")
    ))
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"), p_per_sample=params["p_gamma"]))

    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("advanced_pyramid_augmentations") and not None and params.get("advanced_pyramid_augmentations"):
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                                                                    p_per_sample=0.4,
                                                                    key="data",
                                                                    strel_size=(1, 8)))
            tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                                                                                           key="data",
                                                                                           p_per_sample=0.2,

                                                                                           fill_with_other_class_p=0.0,
                                                                                           dont_do_if_covers_more_than_X_percent=0.15))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    #from batchgenerators.dataloading import SingleThreadedAugmenter
    #batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    #import IPython;IPython.embed()

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'), params.get("num_cached_per_thread"), seeds=seeds_train, pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    #batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads')//2, 1), params.get("num_cached_per_thread"), seeds=seeds_val, pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val

from skimage.morphology import label, ball
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening
import numpy as np
from batchgenerators.transforms import AbstractTransform


class RemoveRandomConnectedComponentFromOneHotEncodingTransform(AbstractTransform):
    def __init__(self, channel_idx, key="data", p_per_sample=0.2, fill_with_other_class_p=0.25,
                 dont_do_if_covers_more_than_X_percent=0.25):
        """
        :param dont_do_if_covers_more_than_X_percent: dont_do_if_covers_more_than_X_percent=0.25 is 25\%!
        :param channel_idx: can be list or int
        :param key:
        """
        self.dont_do_if_covers_more_than_X_percent = dont_do_if_covers_more_than_X_percent
        self.fill_with_other_class_p = fill_with_other_class_p
        self.p_per_sample = p_per_sample
        self.key = key
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in self.channel_idx:
                    workon = np.copy(data[b, c])
                    num_voxels = np.prod(workon.shape)
                    lab, num_comp = label(workon, return_num=True)
                    if num_comp > 0:
                        component_ids = []
                        component_sizes = []
                        for i in range(1, num_comp + 1):
                            component_ids.append(i)
                            component_sizes.append(np.sum(lab == i))
                        component_ids = [i for i, j in zip(component_ids, component_sizes) if j < num_voxels*self.dont_do_if_covers_more_than_X_percent]
                        #_ = component_ids.pop(np.argmax(component_sizes))
                        #else:
                        #    component_ids = list(range(1, num_comp + 1))
                        if len(component_ids) > 0:
                            random_component = np.random.choice(component_ids)
                            data[b, c][lab == random_component] = 0
                            if np.random.uniform() < self.fill_with_other_class_p:
                                other_ch = [i for i in self.channel_idx if i != c]
                                if len(other_ch) > 0:
                                    other_class = np.random.choice(other_ch)
                                    data[b, other_class][lab == random_component] = 1
        data_dict[self.key] = data
        return data_dict


class MoveSegAsOneHotToData(AbstractTransform):
    def __init__(self, channel_id, all_seg_labels, key_origin="seg", key_target="data", remove_from_origin=True):
        self.remove_from_origin = remove_from_origin
        self.all_seg_labels = all_seg_labels
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel_id = channel_id

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)
        seg = origin[:, self.channel_id:self.channel_id+1]
        seg_onehot = np.zeros((seg.shape[0], len(self.all_seg_labels), *seg.shape[2:]), dtype=seg.dtype)
        for i, l in enumerate(self.all_seg_labels):
            seg_onehot[:, i][seg[:, 0] == l] = 1
        target = np.concatenate((target, seg_onehot), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            remaining_channels = [i for i in range(origin.shape[1]) if i != self.channel_id]
            origin = origin[:, remaining_channels]
            data_dict[self.key_origin] = origin
        return data_dict


class ApplyRandomBinaryOperatorTransform(AbstractTransform):
    def __init__(self, channel_idx, p_per_sample=0.3, any_of_these=(binary_dilation, binary_erosion, binary_closing, binary_opening),
                 key="data", strel_size=(1, 10)):
        """

        :param channel_idx: can be list or int
        :param p_per_sample:
        :param any_of_these:
        :param fill_diff_with_other_class:
        :param key:
        :param strel_size:
        """
        self.strel_size = strel_size
        self.key = key
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample

        assert not isinstance(channel_idx, tuple), "bäh"

        if not isinstance(channel_idx, list):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                ch = deepcopy(self.channel_idx)
                np.random.shuffle(ch)
                for c in ch:
                    operation = np.random.choice(self.any_of_these)
                    selem = ball(np.random.uniform(*self.strel_size))
                    workon = np.copy(data[b, c]).astype(int)
                    res = operation(workon, selem).astype(workon.dtype)
                    data[b, c] = res

                    # if class was added, we need to remove it in ALL other channels to keep one hot encoding
                    # properties
                    # we modify data
                    other_ch = [i for i in ch if i != c]
                    if len(other_ch) > 0:
                        was_added_mask = (res - workon) > 0
                        for oc in other_ch:
                            data[b, oc][was_added_mask] = 0
                        # if class was removed, leave it at backgound
        data_dict[self.key] = data
        return data_dict

class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp

    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self, regions, seg_key="seg", output_key="seg", seg_channel=0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r in range(num_regions):
                    for l in self.regions[r]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict
