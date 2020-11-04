#!/usr/bin/env python3
#
# 创建 nnU-Net 所需 dataset.json 文件
#
import json
import os
import os.path as osp
from collections import OrderedDict


if __name__ == '__main__':
    # 注意修改路径
    datadir = "/home/zhangyongtao/PycharmProjects/code/result_2/data/nnUNet_raw/Task04_cancer"
    json_dict = OrderedDict()
    json_dict['name'] = "Stomach"
    json_dict['description'] = "stomach cancer CT segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "NA"
    json_dict['licence'] = "NA"
    json_dict['release'] = "1.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "cancer",
    }

    json_dict['training'] = [{'image': "./imagesTr/{}".format(imgfile),
                              'label': "./labelsTr/{}".format(maskfile)}
                             for imgfile, maskfile in zip(sorted(os.listdir(osp.join(datadir, 'imagesTr'))),
                                                          sorted(os.listdir(osp.join(datadir, 'labelsTr'))))]
    json_dict['test'] = ["./imagesTs/{}".format(imgfile)
                         for imgfile in os.listdir(osp.join(datadir, 'imagesTs'))]
    json_dict['numTraining'] = len(json_dict['training'])
    json_dict['numTest'] = len(json_dict['test'])
    with open(osp.join(datadir, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f)
