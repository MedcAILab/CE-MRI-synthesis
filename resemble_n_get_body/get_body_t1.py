"""
MRI 身体区域获取,最开始的取body对resample文件夹的操作
"""
import SimpleITK as sitk
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import ants
import pandas as pd


def morph_operation(img, kernel_size=(5, 5), anchor=(-1, -1), operation_type=cv2.MORPH_OPEN):
    body_mask = sitk.GetArrayFromImage(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    mask_final = body_mask.copy()

    for i in range(body_mask.shape[0]):
        if np.max(body_mask[i, :, :]) > 0:
            if operation_type == "erode":
                mask_final[i, :, :] = cv2.erode(body_mask[i, :, :], kernel, anchor=anchor, iterations=1)
            elif operation_type == "dilate":
                mask_final[i, :, :] = cv2.dilate(body_mask[i, :, :], kernel, anchor=anchor, iterations=1)
            else:
                mask_final[i, :, :] = cv2.morphologyEx(body_mask[i, :, :], operation_type, kernel, iterations=1)
    mask_final = sitk.GetImageFromArray(mask_final.astype(np.uint8))
    return mask_final


def fill_inter_bone(mask):
    # 对一张图像做孔洞填充，读入的是一层
    mask = mask_fill = mask.astype(np.uint8)
    if np.sum(mask[:]) != 0:  # 即读入图层有值
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)
        mask_fill = sum(contour_list)
        mask_fill[mask_fill >= 1] = 1
    return mask_fill.astype(np.uint8)


def fill_inter_3D(mask, other_axis=True):
    # 对3D图像做孔洞填充，即三个维度的fill_inter_bone
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    mask_final = mask.copy()
    for i in range(mask.shape[0]):
        if np.max(mask[i, :, :]) > 0:
            mask_final[i, :, :] = fill_inter_bone(mask_final[i, :, :])
    if other_axis:
        for i in range(mask.shape[1]):
            if np.max(mask[:, i, :]) > 0:
                mask_final[:, i, :] = fill_inter_bone(mask_final[:, i, :])
        for i in range(mask.shape[2]):
            if np.max(mask[:, :, i]) > 0:
                mask_final[:, :, i] = fill_inter_bone(mask_final[:, :, i])
    return mask_final.astype(np.uint8)


def getmaxcomponent(mask_array, min_size=1e4, check_num=10000, print_num=False, id_num=None):
    """
    获取最大连通域
    :reg_param mask_array: 输入的二值mask image模式
    :reg_param min_size: 最大连通域最小包含的体素数量
    :reg_param check_num: 检查多少个连通域
    :reg_param print_num: 说明有多少连通域
    :return:
    """
    # sitk方法，更快,得到的是相当于ski的connectivity=3的结果
    # 建立报错文件
    error_f = r'/data/nas3/MJY_file/CE-MRI/body_error.txt'
    if isinstance(mask_array, np.ndarray):
        mask_array = sitk.GetImageFromArray(mask_array)
    cca = sitk.ConnectedComponentImageFilter()
    # cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = mask_array
    output_ex = cca.Execute(_input)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    if num <= check_num:
        check_num = num
        # print(check_num)
    # 获得连通域(ski方法)
    # labeled_img, num = ski.measure.label(mask_array, connectivity=3, return_num=True)
    max_label = 1
    max_num = 0
    for i in range(1, check_num + 1):  # 不必全部遍历，一般在前面就有对应的label，减少计算时间
        if np.sum(labeled_img == i) < min_size:  # 小于设置的最小体素数量，直接不计
            continue
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    if print_num:
        print(str(num) + '/' + str(max_label) + ':' + str(np.sum(labeled_img == max_label)))  # 看第几个是最大的
    if np.sum(labeled_img == max_label) < min_size:  # 最终大小还是小于设定值，说明check_num设太小了
        print("Don't get the right component!! size：" + str(np.sum(labeled_img == max_label)))
        if id_num:
            with open(error_f, mode="a+") as f:
                print(id_num, file=f)
            f.close()

    maxcomponent = (labeled_img == max_label).astype(np.uint8)
    maxcomponent = sitk.GetImageFromArray(maxcomponent)
    return maxcomponent


if __name__ == '__main__':
    nii_dir = r'/data/newnas/MJY_file/CE-MRI/test_PCa_data_resample'
    body_save_folder_path = r'/data/newnas/MJY_file/CE-MRI/test_PCa_data_resample'
    if not os.path.exists(body_save_folder_path):
        os.mkdir(body_save_folder_path)
    # 所有文件
    dcms = [item for item in glob.glob(nii_dir + "/*") if os.path.isdir(item)]
    # 待修正的mask
    # dcms = [os.path.join(nii_dir, str(ids)) for ids in refine_id_list]
    # dcms = [os.path.join(nii_dir, "10170732")]
    # dcms = dcms[30:]
    # 要操作的id范围

    for i, dcm in enumerate(dcms, 1):
        id_num = os.path.basename(dcm)
        # if id_num != "0001355518":
        #     continue
        info_dict = {}
        # 获取去除trans_file文件夹的列表
        nii_list = os.listdir(dcm)
        if 'trans_file' in nii_list:
            nii_list.remove("trans_file")
        # if len(nii_list) > 4:
        #     raise "{}more then 4 file".format(dcm)
        for nii_name in nii_list:
            if 'T1.' in nii_name:
                img = sitk.ReadImage(os.path.join(dcm, nii_name))
                ####debugggggggggggggggggggggggggggggggggggg
                # input_image = img
                # # 创建中值滤波器
                # median_filter = sitk.MedianImageFilter()
                # # 设置滤波器参数
                # median_filter.SetRadius(7)
                # # 应用滤波器
                # output_image = median_filter.Execute(input_image)
                # new_img = output_image
                ####debugggggggggggggggggggggggggggggggggggg
                # # 初始化 K-Means 聚类分割器
                # kmeans_filter = sitk.ScalarImageKmeansImageFilter()
                # # 执行聚类分割
                # kmeans_image = kmeans_filter.Execute(output_image)
                ####debugggggggggggggggggggggggggggggggggggg
                # 决定用哪个输出做下一步处理
                img_array = sitk.GetArrayFromImage(img)
                # 取大津法的最佳阈值(只能2d)
                # 2D取阈值法
                # body_machine_mask = np.zeros(img_array.shape)  # 取同样大小的数组
                # for layer in range(img_array.shape[0]):
                #     ret = cv2.threshold(img_array[layer, :, :].astype(np.uint16), 0,
                #                         np.max(img_array), cv2.THRESH_TRUNC + cv2.THRESH_OTSU)[0]
                #     body_machine_mask[layer, :, :] = (img_array[layer, :, :] > ret).astype(np.uint8)  # 该层独立做阈值处理
                ret = 150 #阈值

                # 获取身体mask
                # body_machine_mask = body_machine_mask.astype(np.uint8)
                new_img = (img_array > ret).astype(np.uint8)
                # 填补孔洞
                new_img = fill_inter_3D(new_img)
                # 做一次开运算 去伪影
                new_img = sitk.GetImageFromArray(new_img)
                new_img = morph_operation(new_img, kernel_size=(7, 7), operation_type=cv2.MORPH_OPEN)
                # # body_machine_mask = sitk.BinaryMorphologicalClosing(new_img, (5, 5, 5))
                # 取最大连通域
                new_img = getmaxcomponent(new_img,
                                                min_size=1e4,
                                                check_num=50,
                                                print_num=False,
                                                id_num=None)
                # 填补孔洞
                new_img = fill_inter_3D(new_img)
                # 转sitk操作
                new_img = sitk.GetImageFromArray(new_img)
                # 腐蚀 去掉细伪影
                # new_img = sitk.BinaryErode(new_img, (9, 9, 9))
                new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="erode")
                # # 开运算 多一次去掉细伪影
                # new_img = sitk.BinaryMorphologicalOpening(new_img, (9, 9, 9))
                new_img = morph_operation(new_img, kernel_size=(9, 9), operation_type=cv2.MORPH_OPEN)
                # 去除细伪影后去除小连通域
                new_img = getmaxcomponent(new_img, min_size=1e4, check_num=50, print_num=False)
                # 去除连通域后膨胀再闭运算
                # new_img = sitk.BinaryDilate(new_img, (9, 9, 9))
                # new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type=cv2.MORPH_OPEN)
                new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="dilate")
                # 闭运算 填补可能出现的细节缺漏
                # new_img = sitk.BinaryMorphologicalClosing(new_img, (9, 9, 9))
                new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type=cv2.MORPH_CLOSE)
                # # 膨胀（左右少,上下多，前后更少）获得大一点的mask 提高容错
                # new_img = sitk.BinaryDilate(new_img, (5, 5, 5))
                new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type="dilate")
                # 闭开运算
                new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type=cv2.MORPH_CLOSE)
                new_img = sitk.BinaryMorphologicalOpening(new_img, (3, 3, 3))
                # 转回array再填补孔洞
                # body_mask_max = sitk.GetArrayFromImage(new_img)
                # body_mask_max = fill_inter_3D(body_mask_max)
                # 转回去保存
                # new_img = sitk.GetImageFromArray(body_mask_max)
                # new_img = body_mask_max
                # new_img.CopyInformation(img)
                save_nii = os.path.join(body_save_folder_path, id_num, "body_mask.nii.gz")
                # if not os.path.exists(os.path.dirname(save_nii)):
                #     os.makedirs(os.path.dirname(save_nii))
                sitk.WriteImage(new_img, save_nii)
                # sitk.WriteImage(output_image,os.path.dirname(save_nii)+"/median_f_img.nii")
        print("{} finished".format(dcm), "{}/{}".format(i, len(dcms)))
