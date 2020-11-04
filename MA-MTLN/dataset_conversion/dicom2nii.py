#!/usr/bin/env python3
#
import os
import os.path as osp
import SimpleITK as sitk
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='input dir')
    parser.add_argument('--output', type=str, required=True,
                        help='output dir')

    return parser.parse_args()


def dicomseries2nii(datadir, imagedir, maskdir, patientid):
    """
    将 DICOM Series 数据转换为 nii.gz，输出类似 imagedir/patient-001.nii.gz,
    maskdir/patient-001.nii.gz
    :param datadir: 输入目录
    :param imagedir: 输出的 image 目录
    :param maskdir: 输出的 label 目录
    :return:
    """
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(datadir)
    if not series_IDs:
        print("No DICOM series found in {}, skipping...".format(datadir))
    else:
        # print("converting {} in {}".format(series_IDs, datadir))
        # mask = sitk.ReadImage(osp.join(datadir, '1.nii.gz'))
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(datadir, series_IDs[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        sitk.WriteImage(image, osp.join(imagedir, 'patient-{:0>3d}.nii.gz'.format(int(patientid))))
        sitk.WriteImage(mask, osp.join(maskdir, 'patient-{:0>3d}.nii.gz'.format(int(patientid))))
        print("converting {} to {}".format(datadir, osp.join(imagedir, 'patient-{:0>3d}.nii.gz'.format(int(patientid)))))


if __name__ == '__main__':
    args = get_args()
    if not osp.exists(osp.join(args.output, 'imagesTr')):
        os.makedirs(osp.join(args.output, 'imagesTr'))
    if not osp.exists(osp.join(args.output, 'labelsTr')):
        os.makedirs(osp.join(args.output, "labelsTr"))

    for patientid in os.listdir(args.input):
        if 'china' in args.input:
            dicomseries2nii(osp.join(args.input, patientid, "1/1"),
                      osp.join(args.output, 'imagesTr'),
                      osp.join(args.output, "labelsTr"),
                      patientid)
        else:
            dicomseries2nii(osp.join(args.input, patientid),
                      osp.join(args.output, 'imagesTr'),
                      osp.join(args.output, "labelsTr"),
                      patientid)
