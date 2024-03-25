"""
对nii_data_pre图像做归一化和删除缺块的切片保存到新的文件夹

"""
import glob
import os

import SimpleITK as sitk
import numpy as np


# import h5py


# import matplotlib.pyplot as plt

def find_valid_slice(array, mask):
    """
    non-zero area > 50%
    :return:
    """
    non_zero_array = array > 0
    valid_list = []
    for i in range(array.shape[0]):
        ratio = non_zero_array[i].sum() / (mask[i].sum() + 1)
        if ratio > 0.5:
            valid_list.append(i)
    return np.array(valid_list)


if __name__ == "__main__":
    nii_pre_path = r'/data/newnas/MJY_file/CE-MRI/PCa_new/out_before_norm'
    nii_list = [item for item in glob.glob(nii_pre_path + "/*") if os.path.isdir(item)]
    # nii_list = sorted(nii_list, key=lambda x: int(x.split('/')[-1]))
    mode = "01norm"
    if mode == "stdnorm":
        new_pre_path = r'/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre'
    else:
        new_pre_path = r'/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-01norm/images_ts_out'
    print(new_pre_path)
    # excel_to_delete_slice = r'/data/newnas/MJY_file/CE-MRI/mask处理与信号异常弃用的id信息.xlsx'
    # delelate_id_list = list(pd.read_excel(excel_to_delete_slice, sheet_name="删除的切片信息")["ID"].dropna().astype(str))
    # # nii_list = [os.path.join(nii_pre_path, d_id) for d_id in delelate_id_list]
    # slice_str_list = list(pd.read_excel(excel_to_delete_slice, sheet_name="删除的切片信息")["slice"].dropna().astype(str))
    if not os.path.exists(new_pre_path):
        os.makedirs(new_pre_path)
    temp_list = os.listdir(r'/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre/images_ts_out')
    nii_list = [i for i in nii_list if i.split('/')[-1] in temp_list]
    for idx, nii_folder in enumerate(nii_list):
        id_num = os.path.basename(nii_folder)
        # if idx < 216:
        #     continue
        # if id_num != "P2965982":
        #     continue
        try:
            # 读取图像
            t1_file = os.path.join(nii_folder, "T1.nii.gz")
            t2_file = os.path.join(nii_folder, "T2.nii.gz")
            t1ce_file = os.path.join(nii_folder, "T1CE.nii.gz")
            b50_file = os.path.join(nii_folder, "B50.nii.gz")
            b800_file = os.path.join(nii_folder, "B800.nii.gz")
            b1500_file = os.path.join(nii_folder, "B1400.nii.gz")
            adc_file = os.path.join(nii_folder, "ADC.nii.gz")
            mask_file = os.path.join(nii_folder, "body_mask.nii.gz")
            body_mask = sitk.ReadImage(mask_file)
            body_mask.CopyInformation(sitk.ReadImage(t1_file))
            sitk.WriteImage(body_mask, mask_file)
            file_list = [t1_file, t2_file, t1ce_file, b50_file, b800_file, b1500_file, adc_file, mask_file]
            # t1_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(t1_file)).sum(axis=1).sum(axis=1))
            # t2_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(t2_file)).sum(axis=1).sum(axis=1))
            # t1ce_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(t1ce_file)).sum(axis=1).sum(axis=1))
            # b50_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(b50_file)).sum(axis=1).sum(axis=1))
            # b800_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(b800_file)).sum(axis=1).sum(axis=1))
            # b1500_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(b1500_file)).sum(axis=1).sum(axis=1))
            # adc_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(adc_file)).sum(axis=1).sum(axis=1))
            t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1_file))
            t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_file))
            t1ce_array = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_file))
            b50_array = sitk.GetArrayFromImage(sitk.ReadImage(b50_file))
            b800_array = sitk.GetArrayFromImage(sitk.ReadImage(b800_file))
            b1500_array = sitk.GetArrayFromImage(sitk.ReadImage(b1500_file))
            adc_array = sitk.GetArrayFromImage(sitk.ReadImage(adc_file))
            mask_array = sitk.GetArrayFromImage(body_mask)

            t1_array_se = find_valid_slice(t1_array, mask_array)
            t2_array_se = find_valid_slice(t2_array, mask_array)
            t1ce_array_se = find_valid_slice(t1ce_array, mask_array)
            b50_array_se = find_valid_slice(b50_array, mask_array)
            b800_array_se = find_valid_slice(b800_array, mask_array)
            b1500_array_se = find_valid_slice(b1500_array, mask_array)
            adc_array_se = find_valid_slice(adc_array, mask_array)

            start_point = max([t1_array_se.min(),
                               t2_array_se.min(),
                               t1ce_array_se.min(),
                               b50_array_se.min(),
                               b800_array_se.min(),
                               b1500_array_se.min(),
                               adc_array_se.min()])
            end_point = min([t1_array_se.max(),
                             t2_array_se.max(),
                             t1ce_array_se.max(),
                             b50_array_se.max(),
                             b800_array_se.max(),
                             b1500_array_se.max(),
                             adc_array_se.max()])
            if (end_point - start_point) < 8:
                print(" WARNING: {} slice less than 8".format(id_num))
            for nii in file_list:
                img_o = sitk.ReadImage(nii)
                img = sitk.GetArrayFromImage(img_o)
                if nii != mask_file:
                    if mode == "stdnorm":
                        mean = img[img != 0].mean()
                        std = img[img != 0].std()
                        img = (img - mean) / std
                    else:
                        upper_1onk = img.max() * 0.75
                        img[img > upper_1onk] = upper_1onk
                        img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
                    # img = img * body_mask
                img = img[start_point:end_point + 1]
                img = sitk.GetImageFromArray(img)
                img.SetOrigin(img_o.GetOrigin())
                img.SetSpacing(img_o.GetSpacing())
                img.SetDirection(img_o.GetDirection())
                # 创造路径
                if not os.path.exists(os.path.join(new_pre_path, id_num)):
                    os.makedirs(os.path.join(new_pre_path, id_num))
                if os.path.basename(nii) == "T2-fs.nii.gz":
                    new_nii = os.path.join(new_pre_path, id_num, "T2.nii.gz")
                else:
                    new_nii = os.path.join(new_pre_path, id_num, os.path.basename(nii))
                sitk.WriteImage(img, new_nii)
            print('\r' + str(idx + 1) + '/' + str(len(nii_list)), id_num, end='', flush=True)
        except Exception as e:
            print("\n error in {}, ".format(id_num), e)
