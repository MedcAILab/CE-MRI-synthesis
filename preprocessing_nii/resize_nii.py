import glob
import os.path

import SimpleITK as sitk


# from antspy_registration import get_series


def itk_resample(moving, target, resamplemethod=sitk.sitkLinear):
    # 读取原始图像的spacing和size
    ori_size = moving.GetSize()
    ori_spacing = moving.GetSpacing()
    # 读取重采样的参数
    target_Spacing = target.GetSpacing()
    # 方向和origin不必变动
    ori_direction = moving.GetDirection()
    ori_origin = moving.GetOrigin()
    target_Size = target.GetSize()
    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(-1)
    resampler.SetReferenceImage(target)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize((target_Size[0], target_Size[1], ori_size[2]))
    resampler.SetOutputDirection(ori_direction)
    resampler.SetOutputOrigin(ori_origin)
    resampler.SetOutputSpacing((target_Spacing[0], target_Spacing[1], ori_spacing[2]))
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值是用于PET/CT/MRI之类的，所以保存float32格式
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # 3, sitk.sitkIdentity这个参数的用处还不确定
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(moving)  # 得到重新采样后的图像
    return itk_img_resampled


if __name__ == '__main__':
    dcm_data_dir = r'/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-01norm/images_ts'  # 原始nii文件保存路径
    save_folder = r'/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts'  # 保存路径
    dcms = [item for item in glob.glob(dcm_data_dir + "/*") if os.path.isdir(item)]
    dcms = sorted(dcms)
    # dcms = dcms[68:]
    param_file = r'reg_param/series_param.json'
    # 获取一个内部平扫t1的模板图像，把外部的T1全部统一到一个大小，确保是320*320
    globle_templete = r'/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-01norm/images_tr/0000332968/T1.nii.gz'
    # # itk方法
    globle_templete_img_itk = sitk.ReadImage(globle_templete)
    # 中途重新找回的list
    # tmpp_list = ["10320753","10408044","10424743","10440673"]
    for i, dcm in enumerate(dcms, 1):
        # if os.path.basename(dcm) != "0001182438":
        #     continue
        # get this id 's series_dict
        ser_dict = ["T1CE_body_mask"]
        # set resample dir
        dir_temp = dcm.split('/')
        dir_temp[-3] = os.path.dirname(save_folder).split("/")[-1]
        resample_dir = '/'.join(dir_temp)
        # make new dir
        if not os.path.exists(resample_dir):
            os.makedirs(resample_dir)
        for ser in ser_dict:
            # 重采样t1的
            # 模板
            target_image = globle_templete_img_itk
            # 源序列
            moving_file = os.path.join(dcm, ser + ".nii.gz")
            # moving_image = ants.image_read(moving_file)
            moving_image = sitk.ReadImage(moving_file)
            resampled_img = itk_resample(moving_image, target_image)
            save_nii = os.path.join(resample_dir, os.path.basename(moving_file))
            sitk.WriteImage(resampled_img, save_nii)
        print('{} is done'.format(dcm), '{}/{}'.format(i, len(dcms)))
