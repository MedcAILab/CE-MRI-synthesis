"""
记得把tr和ts都要转换
"""

import os.path

import SimpleITK as sitk
import h5py

if __name__ == "__main__":
    data_dir = "/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre/images_tr"
    save_2d_dir = "/data/newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre/images_tr"
    if not os.path.exists(save_2d_dir):
        os.makedirs(save_2d_dir)
    id_list = os.listdir(data_dir)
    for idx, id_num in enumerate(id_list):
        # if id_num != "0002245347":
        #     continue
        t1 = os.path.join(data_dir, id_num, "T1.nii.gz")
        t2 = os.path.join(data_dir, id_num, "T2.nii.gz")
        b800 = os.path.join(data_dir, id_num, "B800.nii.gz")
        b50 = os.path.join(data_dir, id_num, "B50.nii.gz")
        mask = os.path.join(data_dir, id_num, "body_mask.nii.gz")
        t1ce = os.path.join(data_dir, id_num, "T1CE.nii.gz")
        b1500 = os.path.join(data_dir, id_num, "B1400.nii.gz")
        adc = os.path.join(data_dir, id_num, "ADC.nii.gz")
        pro_mask = os.path.join(data_dir, id_num, "prostate_mask.nii.gz")

        # 读取3d图像
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask, outputPixelType=sitk.sitkUInt8))
        slice_num = mask_array.shape[0]

        t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1, outputPixelType=sitk.sitkFloat32))
        t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2, outputPixelType=sitk.sitkFloat32))
        b800_array = sitk.GetArrayFromImage(sitk.ReadImage(b800, outputPixelType=sitk.sitkFloat32))
        b50_array = sitk.GetArrayFromImage(sitk.ReadImage(b50, outputPixelType=sitk.sitkFloat32))
        t1ce_array = sitk.GetArrayFromImage(sitk.ReadImage(t1ce, outputPixelType=sitk.sitkFloat32))
        b1500_array = sitk.GetArrayFromImage(sitk.ReadImage(b1500, outputPixelType=sitk.sitkFloat32))
        adc_array = sitk.GetArrayFromImage(sitk.ReadImage(adc, outputPixelType=sitk.sitkFloat32))
        pro_mask_array = sitk.GetArrayFromImage(sitk.ReadImage(pro_mask, outputPixelType=sitk.sitkUInt8))
        # =====================================================
        for idj, layer in enumerate(range(slice_num)):
            h5_file_path = os.path.join(save_2d_dir, id_num, "layer_{}.h5".format(layer))
            if not os.path.exists(os.path.dirname(h5_file_path)):
                os.makedirs(os.path.dirname(h5_file_path))
            with h5py.File(h5_file_path, "w") as h5_file:
                h5_file["t1"] = t1_array[layer, :, :]
                h5_file["t2"] = t2_array[layer, :, :]
                h5_file["b800"] = b800_array[layer, :, :]
                h5_file["b50"] = b50_array[layer, :, :]
                h5_file["mask"] = mask_array[layer, :, :]
                h5_file["t1ce"] = t1ce_array[layer, :, :]
                h5_file["b1500"] = b1500_array[layer, :, :]
                h5_file["adc"] = adc_array[layer, :, :]
                h5_file["prostate_mask"] = pro_mask_array[layer, :, :]
                h5_file.close()
            print("\r", idj, "/", slice_num, end='')
        print(' ')
        print(idx, "/", len(id_list))
