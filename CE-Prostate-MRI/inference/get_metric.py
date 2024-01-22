import os.path
import sys

import SimpleITK as sitk
import pandas as pd

from inference.test_metrics import *
from inference.test_param import config

if __name__ == "__main__":
    # ==========path============
    dir_prefix = sys.argv[0].split("/newnas")[0]
    config.filepath_img = os.path.join(dir_prefix, config.filepath_img)
    config.result_path = os.path.join(dir_prefix, config.result_path)
    Task_name = config.Task_name
    task_id = config.Task_id
    fold_idx = config.fold_idx
    ckpt_name = config.ckpt_name
    net_mode = config.net_mode
    # ===============model setting==============
    task_name = "{}_{}_{}_fold5-{}".format(Task_name, task_id, net_mode, fold_idx)
    result_path = config.result_path
    gt_dir = os.path.join(config.filepath_img, "images_ts")
    pred_dir = os.path.join(result_path, task_name, "pred_nii_" + ckpt_name.split(".")[0])
    excel_save_dir = os.path.join(result_path, task_name, "pred_nii_" + ckpt_name.split(".")[0] + "_metric.xlsx")
    print("====================={}=======================".format(pred_dir))
    # ============================================
    pred_list = os.listdir(pred_dir)
    metrics = []
    mean_nrmse_metric = 0
    mean_smape_metric = 0
    mean_logac_matric = 0
    mean_medsymac_matric = 0
    for idx, filename in enumerate(pred_list):
        id_num = filename.split("_")[0]
        gt_file = os.path.join(gt_dir, id_num, "T1CE.nii.gz")
        pred_file = os.path.join(pred_dir, filename)
        mask_file = os.path.join(gt_dir, id_num, "body_mask.nii.gz")
        gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
        pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        mask_img = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        # get error matric
        # normalized root mean square error
        nrmse_metric = nrmse(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # symmetric mean absolute percent error
        smape_metric = smape(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # log accuracy ratio
        logac_matric = logac(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # median symmetric accuracy
        medsymac_matric = medsymac(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # get sim metrics
        # =======neighborhood cross correlation=======
        cc_metric = cc_py(gt_file, pred_file, mask_file)

        # =======histogram mutual information=======
        mi_metric = mi_py(gt_file, pred_file, mask_file)
        # mi_metric = nmi(true_array=gt_img,pred_array=pred_img,mask=mask_img)
        # =======ssim=======
        ssim_metric = ssim_torch(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # =======psnr=======
        # psnr_metric = psnr(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # =======lpips=======
        # lpips=lpips_metric(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # lpips = 0
        lpips = med_lpips_metric(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        # =======fid=======
        # fid = fid_torch(true_array=gt_img, pred_array=pred_img, mask=mask_img, compute=(idx == (len(pred_list) - 1)))
        fid = 0
        # =======vif=======
        # vif = vif_torch(true_array=gt_img, pred_array=pred_img, mask=mask_img)
        vif = 0
        all_metric = [str(id_num), nrmse_metric, smape_metric, logac_matric, medsymac_matric, cc_metric, mi_metric,
                      ssim_metric, lpips, fid, vif]
        metrics.append(all_metric)
        print("{}/{} {}".format(idx + 1, len(pred_list), id_num),
              "nrmse:{}, smape:{}, logza:{}, medsymaz:{}, cc:{}, mi:{}, ssim:{}, lpips:{}, fid:{}, vif:{}".format(
                  *all_metric[1:]))
    # 求平均
    np_vresion = np.array(metrics)[:, 1:].astype(np.float32)
    mean = np.mean(np_vresion, axis=0)
    mean_metric = [0, *mean[:]]
    print("nrmse:{}, smape:{}, logza:{}, medsymaz:{}, cc:{}, mi:{}, ssim:{}, lpips:{}, fid:{}, vif:{}".format(
        *mean_metric[1:]))
    metrics.insert(0, mean_metric)
    # 保存表格
    result = pd.DataFrame(metrics)
    record = pd.ExcelWriter(excel_save_dir, mode='w')
    header = ['ids'] + ["nrmse", "smape", "logac", "medsymac", "cc", "mi", "ssim", "lpips", "fid", "vif"]
    result.to_excel(record, header=header, index=False)
    record.close()
