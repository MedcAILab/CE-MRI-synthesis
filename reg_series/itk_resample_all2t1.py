import glob
import os.path

import SimpleITK as sitk

from antspy_registration import get_series


def itk_resample(moving, target, resamplemethod=sitk.sitkLinear):
    # 初始化一个列表
    target_Size = [0, 0, 0]
    # 读取原始图像的spacing和size
    ori_size = moving.GetSize()
    ori_spacing = moving.GetSpacing()
    # 读取重采样的参数
    target_Spacing = target.GetSpacing()
    # 方向和origin不必变动
    target_direction = target.GetDirection()
    target_origin = target.GetOrigin()
    # 获取重采样的图像大小
    # target_Size[0] = round(ori_size[0] * ori_spacing[0] / target_Spacing[0])
    # target_Size[1] = round(ori_size[1] * ori_spacing[1] / target_Spacing[1])
    # target_Size[2] = round(ori_size[2] * ori_spacing[2] / target_Spacing[2])
    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    # resampler.SetSize(target_Size)
    # resampler.SetOutputDirection(target_direction)
    # resampler.SetOutputOrigin(target_origin)
    # resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值是用于PET/CT/MRI之类的，所以保存float32格式
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # 3, sitk.sitkIdentity这个参数的用处还不确定
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(moving)  # 得到重新采样后的图像
    return itk_img_resampled


if __name__ == '__main__':
    dcm_data_dir = r'/data/newnas/MJY_file/CE-MRI/test_PCa_raw_data'  # 原始nii文件保存路径
    save_folder = r'/data/newnas/MJY_file/CE-MRI/test_PCa_data_resample'  # 保存路径
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # excel = r'/data/newnas/MJY_file/CE-MRI/check_192.xlsx'
    # to_change_id_list = list(pd.read_excel(excel, sheet_name="Sheet1")["ID"].astype(str))
    # front = list(pd.read_excel(excel, sheet_name="Sheet1")["b50头"].astype(int))
    # tail = list(pd.read_excel(excel, sheet_name="Sheet1")["b50尾"].astype(int))
    # slice_dict = {}
    # for id_num,front_num,tail_num in zip(to_change_id_list,front,tail):
    #     slice_dict.update({id_num:slice(front_num-1,tail_num)})
    dcms = [item for item in glob.glob(dcm_data_dir + "/*") if os.path.isdir(item)]

    # dcms = dcms[68:]
    param_file = r'reg_param/series_param.json'
    # 获取一个平扫t1的模板图像，全部统一到一个spacing，[512,512,32]
    # globle_templete = r'/data/newnas/MJY_file/CE-MRI/CE-MRI/10170732/t1tsetracCHENQIWEIE-ACHENQIWEIs005a1001.nii'
    # globle_templete_img = nib.load(globle_templete)
    # globle_templete_img = ants.from_nibabel(globle_templete_img)
    # # itk方法
    # globle_templete_img_itk = sitk.ReadImage(globle_templete)
    # 中途重新找回的list
    # tmpp_list = ["10320753","10408044","10424743","10440673"]
    for i, dcm in enumerate(dcms, 1):
        id_name = os.path.basename(dcm)
        # if id_name != "0006092362":
        #     continue
        # get this id 's series_dict
        ser_dict = get_series(dcm, param_file, after_resample_t1=False)

        # set resample dir!!!
        dir_temp = dcm.split('/')
        dir_temp[-2] = os.path.basename(save_folder)
        resample_dir = '/'.join(dir_temp)
        # 模板
        target = os.path.join(dcm_data_dir, dcm, os.path.basename(ser_dict["T1"]["filename"]))
        target_image = sitk.ReadImage(target)
        # make new dir
        if not os.path.exists(resample_dir):
            os.makedirs(resample_dir)
        for ser in ser_dict.keys():
            # 重采样
            if ser in [
                "ADC", "B50", "B1400", "B800", "T2",
                "T1CE",
                # "T2-fs"
            ]:
                if ser_dict[ser]["filename"] != "None":
                    # 目标序列
                    # target_series = ser_dict[ser]["reg_target"]
                    # target_file = ser_dict[target_series]["filename"]
                    # target_image = ants.image_read(target_file)
                    # 模板
                    # target_image = target_image
                    # 源序列
                    moving_file = ser_dict[ser]["filename"]
                    # moving_image = ants.image_read(moving_file)
                    moving_image = sitk.ReadImage(moving_file)
                    # 去除人工筛查的问题片
                    # if id_name in slice_dict.keys():
                    #     moving_array = sitk.GetArrayFromImage(moving_image)
                    #     # 有问题切片变全黑
                    #     zero_array = np.zeros_like(moving_array)
                    #     zero_array[slice_dict[id_name]]=moving_array[slice_dict[id_name]]
                    #     moving_array = zero_array
                    #     # 得到新的itk image
                    #     new_moving_image = sitk.GetImageFromArray(moving_array)
                    #     new_moving_image.CopyInformation(moving_image)
                    #     moving_image = new_moving_image
                    # 重采样图像
                    # resampled_img = ants.resample_image_to_target(image=moving_image,
                    #                                               target=target_image)
                    # resampled_img = ants.to_nibabel(resampled_img)
                    resampled_img = itk_resample(moving_image, target_image)
                    save_nii = os.path.join(resample_dir, ser + ".nii.gz")
                    sitk.WriteImage(resampled_img, save_nii)
        sitk.WriteImage(target_image, os.path.join(resample_dir, "T1.nii.gz"))
        print('{} is done'.format(ser_dict['info']['id']), '{}/{}'.format(i, len(dcms)))
