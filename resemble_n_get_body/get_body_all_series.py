"""
使用resample后的文件做取body的操作！！！
"""
import os
import shutil
import SimpleITK as sitk
import pandas as pd


def main():
    dst_path = r'/dat'
    save_path = r'/dat'
    file_list = os.listdir(dst_path)
    for idx, i in enumerate(file_list):
        # if i != "0006092362":
        #     continue
        try:
            folder = os.path.join(dst_path, i)
            save_folder = os.path.join(save_path, i)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            # 获取body
            body = sitk.ReadImage(os.path.join(folder, "body_mask.nii.gz"))
            body_mask = sitk.GetArrayFromImage(body)

            for filename in os.listdir(folder):
                if filename == 'trans_file':
                    continue
                if filename != "body_mask.nii.gz":
                    img = sitk.ReadImage(os.path.join(folder, filename))
                    img_array = sitk.GetArrayFromImage(img)
                    img_array = img_array * body_mask
                    img_new = sitk.GetImageFromArray(img_array)
                    # 复制原来的元数据
                    img_new.CopyInformation(img)
                    sitk.WriteImage(img_new, os.path.join(save_folder, filename))
            sitk.WriteImage(body, os.path.join(save_folder, "body_mask.nii.gz"))
            print('\r' + str(idx + 1) + '/' + str(len(file_list)), i, end='', flush=True)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
