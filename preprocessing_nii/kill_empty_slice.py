import glob
import os.path

import SimpleITK as sitk
import numpy as np

if __name__ == '__main__':
    dcm_data_dir = r''  # 原始nii文件保存路径
    save_folder = r''  # 保存路径
    body_mask_folder = r''
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    excel = r'/data/newnas/MJY_file/CE-MRI/check_192.xlsx'
    dcms = [item for item in glob.glob(dcm_data_dir + "/*") if os.path.isdir(item)]
    for i, dcm in enumerate(dcms, 1):
        id_num = os.path.basename(dcm)
        t1_file = os.path.join(dcm, "T1_w.nii.gz")
        t2_file = os.path.join(dcm, "T2_w.nii.gz")
        t1ce_file = os.path.join(dcm, "T1CE_w.nii.gz")
        b50_file = os.path.join(dcm, "B50_w.nii.gz")
        b800_file = os.path.join(dcm, "B800_w.nii.gz")
        b1500_file = os.path.join(dcm, "B1400_w.nii.gz")
        adc_file = os.path.join(dcm, "ADC_w.nii.gz")
        mask_file = os.path.join(body_mask_folder, id_num, "body_mask.nii.gz")
        t2_mask_file = os.path.join(body_mask_folder, id_num, "t2_fs_body_mask.nii.gz")
        # t2
        body_mask = sitk.ReadImage(mask_file)
        body_mask.CopyInformation(sitk.ReadImage(t1_file))
        sitk.WriteImage(body_mask, mask_file)
        file_list = [t1_file, t2_file, t1ce_file, b50_file, b800_file, b1500_file, adc_file, mask_file]
        t1_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(t1_file)).sum(1).sum(1))
        t2_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(t2_file)).sum(1).sum(1))
        t1ce_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(t1ce_file)).sum(1).sum(1))
        b50_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(b50_file)).sum(1).sum(1))
        b800_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(b800_file)).sum(1).sum(1))
        b1500_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(b1500_file)).sum(1).sum(1))
        adc_array_se = np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(adc_file)).sum(1).sum(1))
        start_point = max([t1_array_se[0].min(),
                           t2_array_se[0].min(),
                           t1ce_array_se[0].min(),
                           b50_array_se[0].min(),
                           b800_array_se[0].min(),
                           b1500_array_se[0].min(),
                           adc_array_se[0].min()])
        end_point = min([t1_array_se[0].max(),
                         t2_array_se[0].max(),
                         t1ce_array_se[0].max(),
                         b50_array_se[0].max(),
                         b800_array_se[0].max(),
                         b1500_array_se[0].max(),
                         adc_array_se[0].max()])
        for nii in file_list:
            if os.path.basename(nii) != "body_mask.nii.gz":
                continue
            img_o = sitk.ReadImage(nii)
            img = sitk.GetArrayFromImage(img_o)
            img = img[start_point:end_point + 1]
            img = sitk.GetImageFromArray(img)
            img.SetOrigin(img_o.GetOrigin())
            img.SetSpacing(img_o.GetSpacing())
            img.SetDirection(img_o.GetDirection())
            if not os.path.exists(os.path.join(save_folder, id_num)):
                os.makedirs(os.path.join(save_folder, id_num))
            new_name = os.path.basename(nii).split(".")[0] + ".nii.gz"
            new_nii = os.path.join(save_folder, id_num, new_name)
            sitk.WriteImage(img, new_nii)
            # os.remove(os.path.join(save_folder,id_num,"body_ma.nii.gz"))
            print('\r' + str(i) + '/' + str(len(dcms)), id_num, end='', flush=True)
