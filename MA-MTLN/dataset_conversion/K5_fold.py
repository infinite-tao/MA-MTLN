#!/usr/bin/env python
#
# 五折交叉验证的划分，生成 splits_final.pkl 文件
# 用于五折交叉验证数据读取
# 这里主要是用 GroupKFold 来保证同一个病人的数据只会出现一个 fold 内
#
import os
import os.path as osp
import pickle
from collections import OrderedDict

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder


if __name__ == '__main__':
    datadir = '/home/zhangyongtao/PycharmProjects/data/preprocessed/nnU-Net-formats'
    group1_data = os.listdir(osp.join(datadir, 'Task02_Shanxi', 'imagesTr'))
    group2_data = os.listdir(osp.join(datadir, 'Task03_Shaanxi', 'imagesTr'))
    group3_data = os.listdir(osp.join(datadir, 'Task01_ChinaJapan', 'imagesTr'))

    group1_data = [item.replace('.nii.gz', '') for item in group1_data]
    group2_data = [item.replace('.nii.gz', '') for item in group2_data]
    group3_data = [item.replace('.nii.gz', '') for item in group3_data]
    groups_data = group1_data + group2_data + group3_data
    groups_data = sorted(groups_data)
    # 同一个病人的数据文件名差别在 _ 之后的那部分
    # 删掉，然后用 OrdinalEncoder 编码一下
    # groups 和 groups_data 对应起来，groups 中的数字表示组别
    # Shaanxi 的数据一个病人有多组数据，需要事先重命名，加上 _ 分隔的后缀
    # eg. Shaanxi-patient-19_a19.nii.gz, Shaanxi-patient-19_d19.nii.gz
    # 这里假定只有一个病人多组数据的时候文件名才会有一个下划线，直接用下划线区分
    groups_data_tmp = [item.split('_')[0] for item in groups_data]
    groups_data_tmp = np.array(groups_data_tmp)
    enc = OrdinalEncoder(dtype=np.uint8)
    groups = enc.fit_transform(groups_data_tmp.reshape(-1, 1))
    groups = groups.flatten()
    groups_data = np.array(groups_data)
    # 按组分成5折，同一个病人的数据分在同一组
    kfold = GroupKFold(n_splits=5)
    fold_data = OrderedDict()
    result = []
    i = 0
    for train, val in kfold.split(X=groups_data, groups=groups):
        fold_data['train'] = groups_data[train]
        fold_data['val'] = groups_data[val]
        result.append(fold_data)
        result.append(fold_data.copy())
    # save_pickle(result, "./splits_final.pkl")
    with open('splits_final.pkl', 'wb') as f:
        pickle.dump(fold_data, f)
        print(result)


