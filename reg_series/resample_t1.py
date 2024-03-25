import glob
import os.path

import SimpleITK as sitk


def itk_resample(moving, target, resamplemethod=sitk.sitkLinear):
    # 初始化一个列表
    target_Size = [0, 0, 0]
    # 读取原始图像的spacing和size
    ori_size = moving.GetSize()
    ori_spacing = moving.GetSpacing()
    # 读取重采样的参数
    target_Spacing = target.GetSpacing()
    # 方向和origin不必变动
    target_direction = moving.GetDirection()
    target_origin = moving.GetOrigin()
    # 获取重采样的图像大小
    target_Size[0] = round(ori_size[0] * ori_spacing[0] / target_Spacing[0])
    target_Size[1] = round(ori_size[1] * ori_spacing[1] / target_Spacing[1])
    target_Size[2] = round(ori_size[2] * ori_spacing[2] / target_Spacing[2])
    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(target)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值是用于PET/CT/MRI之类的，所以保存float32格式
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # 3, sitk.sitkIdentity这个参数的用处还不确定
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(moving)  # 得到重新采样后的图像
    return itk_img_resampled


if __name__ == '__main__':
    dcm_data_dir = r'/data/newnas/MJY_file/CE-MRI/CE-MRI-PCa'  # 原始nii文件保存路径
    save_folder = r'/data/newnas/MJY_file/CE-MRI/CE-MRI'  # 保存路径
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    excel = r'/data/newnas/MJY_file/CE-MRI/res_check.xlsx'
    dcms = [item for item in glob.glob(dcm_data_dir + "/*") if os.path.isdir(item)]
    dcms = sorted(dcms, key=lambda x: int(x.split('/')[-1]))
    # dcms = dcms[68:]
    param_file = r'reg_param/series_param.json'
    # 获取一个平扫t1的模板图像，全部统一到一个spacing，[512,512,32]
    globle_templete = r'/data/newnas/MJY_file/CE-MRI/CE-MRI/10170732/t1tsetracCHENQIWEIE-ACHENQIWEIs005a1001.nii'
    # # itk方法
    globle_templete_img_itk = sitk.ReadImage(globle_templete)
    # 中途重新找回的list
    # tmpp_list = ["10320753","10408044","10424743","10440673"]
    for i, dcm in enumerate(dcms, 1):
        # if os.path.basename(dcm) != "10336220":
        #     continue
        # get this id 's series_dict
        ser_dict = get_series(dcm, param_file)
        # set resample dir
        dir_temp = dcm.split('/')
        dir_temp[-2] = "CE-MRI-resample"
        resample_dir = '/'.join(dir_temp)
        # make new dir
        if not os.path.exists(resample_dir):
            os.makedirs(resample_dir)
        for ser in ser_dict.keys():
            # 重采样t1的
            if ser == "T1":
                if ser_dict[ser]["filename"] != "None":
                    # 模板
                    target_image = globle_templete_img_itk
                    # 源序列
                    moving_file = ser_dict[ser]["filename"]
                    # moving_image = ants.image_read(moving_file)
                    moving_image = sitk.ReadImage(moving_file)
                    resampled_img = itk_resample(moving_image, target_image)
                    save_nii = os.path.join(resample_dir, os.path.basename(moving_file))
                    sitk.WriteImage(resampled_img, save_nii)
        print('{} is done'.format(ser_dict['info']['id']), '{}/{}'.format(i, len(dcms)))
