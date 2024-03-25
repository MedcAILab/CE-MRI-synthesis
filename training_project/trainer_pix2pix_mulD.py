# -*- coding: utf-8 -*-
"""
@Project ：CE-Prostate-MRI
@File    ：trainer_pix2pix_mulD.py
@IDE     ：PyCharm
@Author  ：MJY
@Date    ：2024/1/22 14:48
"""

import os
import time

import SimpleITK as sitk
import lightning.pytorch as pl
import numpy as np
import torch
from monai.data import Dataset, CacheDataset, decollate_batch, DataLoader, pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import MAEMetric, SSIMMetric
from net.pix2pix_HD_model.networks import define_D
from segmentation_models_pytorch import Unet
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from loss_function.losses_function import loss_picker
from training_project.custom_transform import *
from training_project.utils.get_dist_loss import distance_loss
from training_project.utils.image_pool import ImagePool
from training_project.utils.progress_bar import printProgressBar
from training_project.utils.save_tensor_img import tensor2im


def get_duration_time_str(s_time, e_time):
    h, remainder = divmod((e_time - s_time), 3600)  # 小时和余数
    m, s = divmod(remainder, 60)  # 分钟和秒
    time_str = "%02d h:%02d m:%02d s" % (h, m, s)
    return time_str


class Pix2Pix_2d_MulD(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.automatic_optimization = False
        # =============================dataset===================================
        self.val_ds = None
        self.train_ds = None
        self.test_ds = None
        self.num_workers = config.num_workers
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size
        self.num_samples = config.num_samples
        self.val_num_samples = config.val_num_samples
        self.dataset_type = config.dataset_type
        self.val_transforms = None
        self.train_transforms = None
        self.test_transforms = None
        self.crop_size = config.crop_size
        # =============================fold======================================
        self.fold_K = config.fold_K
        self.fold_idx = config.fold_idx
        # ================================net====================================
        input_channels = len(config.train_keys)
        output_channel = 1
        self.net_G = Unet(encoder_name='timm-regnety_160',
                          encoder_weights='imagenet',
                          encoder_depth=4,
                          decoder_channels=[128, 64, 32, 16],
                          # decoder_attention_type="scse",
                          decoder_use_batchnorm=True,
                          in_channels=input_channels, classes=1)
        # self.net_G = attention_unet
        self.net_D = define_D(input_nc=input_channels + output_channel,
                              ndf=64,
                              n_layers_D=config.n_layers_D,
                              norm="instance",
                              use_sigmoid=False,
                              num_D=config.num_D,
                              getIntermFeat=True if "G_Feat_loss" in config.loss_weight_dict.keys() else False)
        # ================================loss&metric============================
        self.criterion_dict = None
        self.loss_weight_dict = config.loss_weight_dict
        self.mae_metric = MAEMetric(reduction="mean", get_not_nans=False)
        self.ssim_metric = SSIMMetric(spatial_dims=2, win_size=9)

        self.best_val_mae = 1000
        self.best_val_ssim = 0
        self.best_val_epoch = 0
        # =================================optimiser======================================
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # ====================================lr===========================================
        self.max_lr = config.lr
        self.min_lr = config.lr_low
        self.lr_list = {"lr": []}
        # =============================training setting=====================================
        self.random_state = config.seed
        self.random_prob = config.augmentation_prob
        self.max_epochs = config.num_epochs
        self.warmup_epochs = config.lr_warm_epoch
        self.cos_epochs = config.lr_cos_epoch
        # self.metric_values = {"MAE": []}
        self.epoch_loss_values = []
        # self.training_loss = {"loss": [], "MSEloss": [], "ms_ssim_loss": [], "monai_ssmi_loss": []}
        # self.training_ms_ssim = {"ms_ssim_loss": []}
        self.fake_pool = ImagePool(config.pool_size)
        # =============================training variable====================================
        self.train_s_time = 0
        self.train_e_time = 0
        self.val_s_time = 0
        self.val_e_time = 0
        self.predict_tic = None
        self.predict_toc = None
        self.keys = config.train_keys
        # =============================文件地址=======================================
        self.data_dir = config.h5_2d_img_dir
        self.train_dir = os.path.join(self.data_dir, "images_tr")
        self.test_dir = os.path.join(self.data_dir, "images_ts")
        self.template_dir = config.filepath_img
        self.log_pic_dir = os.path.join(config.root_dir, "loss_pic")
        self.result_dir = config.root_dir
        self.record_file = os.path.join(config.root_dir, "log_txt.txt")
        self.pred_result_dir = os.path.join(self.result_dir, "pred_nii")
        # if not os.path.exists(self.pred_result_dir):
        #     os.makedirs(self.pred_result_dir)
        if not os.path.exists(self.log_pic_dir):
            os.makedirs(self.log_pic_dir)
        # 初始化 training output saving
        # self.training_step_output = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.net_G(x)

    def print_to_txt(self, *args):
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def do_split(self, K, fold):
        """
        :reg_param K: 分几折
        :reg_param fold: 第几折，从1开始
        :return:分折的病人id列表[train,val]
        """
        fold_train = []
        fold_test = []

        kf = KFold(n_splits=K, random_state=self.random_state, shuffle=True)
        id_list = sorted(os.listdir(self.train_dir))
        for train_index, test_index in kf.split(id_list):
            fold_train.append(np.array(id_list)[train_index])
            fold_test.append(np.array(id_list)[test_index])

        train_id = fold_train[fold - 1]
        test_id = fold_test[fold - 1]
        self.print_to_txt(f'train_id:{len(train_id)}||valid_id:{len(test_id)}')
        return [train_id, test_id]

    def get_data_dict(self, id_list):
        # 输入id的list获取数据字典
        data_dict = []
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(self.train_dir, id_num)))
            for layer in layer_list:  # 头尾不要?
                new_data_dict = {"path": os.path.join(self.train_dir, id_num, layer)}
                data_dict.append(new_data_dict)
        return data_dict

    def get_test_data_dict(self):
        # 输入test_id的list获取数据字典
        data_dict = []
        id_list = sorted(os.listdir(self.test_dir))
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(self.test_dir, id_num)))
            for layer in layer_list:
                new_data_dict = {"path": os.path.join(self.test_dir, id_num, layer)}
                data_dict.append(new_data_dict)
        return data_dict

    def get_dataset(self, data_list, transform, mode="train", dataset_type="normal"):
        """
        :param data_list:
        :param transform:
        :param mode: "train" or "val"
        :param dataset_type: "normal" or "cache"
        :return:
        """
        if mode == "train":
            if dataset_type == "normal":
                self.train_ds = Dataset(
                    data=data_list,
                    transform=transform,
                )
            elif dataset_type == "cache":
                self.train_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=300,
                    cache_rate=0.6,
                    num_workers=self.num_workers,
                )
        elif mode == "val":
            if dataset_type == "normal":
                self.val_ds = Dataset(
                    data=data_list,
                    transform=transform
                )
            elif dataset_type == "cache":
                self.val_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=100,
                    cache_rate=1,
                    num_workers=self.num_workers,
                )
        elif mode == "test":
            if dataset_type == "normal":
                self.test_ds = Dataset(
                    data=data_list,
                    transform=transform
                )
            elif dataset_type == "cache":
                self.test_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=100,
                    cache_rate=1,
                    num_workers=self.num_workers,
                )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_mae"] = self.best_val_mae
        checkpoint["best_metric"] = self.best_val_ssim
        checkpoint["best_val_epoch"] = self.best_val_epoch
        checkpoint["criterion_dict"] = self.criterion_dict

    def on_load_checkpoint(self, checkpoint):
        self.best_val_mae = checkpoint["best_mae"]
        self.best_val_ssim = checkpoint["best_metric"]
        self.best_val_epoch = checkpoint["best_val_epoch"]
        self.criterion_dict = checkpoint["criterion_dict"]

    def prepare_data(self):
        # prepare data
        # 根据dir 获取train：0 val：1
        print("preparing data with val")
        # datasets = sorted(os.listdir(self.train_dir))
        datasets = self.do_split(self.fold_K, self.fold_idx)
        train_dict = self.get_data_dict(datasets[0])
        val_dict = self.get_data_dict(datasets[1])
        test_dict = self.get_test_data_dict()
        self.train_transforms = get_2d_train_transform(keys=self.keys, random_prob=self.random_prob)
        self.val_transforms = get_2d_val_transform(keys=self.keys)
        self.test_transforms = get_2d_test_transform(keys=self.keys)
        # 获取dataset 方法内直接赋值self.train_ds, self.val_ds, self.test_ds
        self.get_dataset(train_dict, self.train_transforms, mode="train", dataset_type=self.dataset_type)
        self.get_dataset(val_dict, self.val_transforms, mode="val", dataset_type=self.dataset_type)
        self.get_dataset(test_dict, self.test_transforms, mode="test", dataset_type=self.dataset_type)
        # 搞搞loss

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return val_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return pred_loader

    def configure_losses(self):
        self.criterion_dict = {}
        for loss_name, weight in self.loss_weight_dict.items():
            self.criterion_dict.update({loss_name: loss_picker(loss_name)})
        self.print_to_txt("loss&weight |", self.loss_weight_dict)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.net_G.parameters(), betas=(self.beta1, self.beta2), lr=self.max_lr)
        optimizer_d = torch.optim.Adam(self.net_D.parameters(), betas=(self.beta1, self.beta2), lr=self.max_lr)
        return ({"optimizer": optimizer_g,
                 "lr_scheduler": {"scheduler": CosineAnnealingLR(optimizer_g, self.max_epochs, eta_min=self.min_lr),
                                  "interval": "epoch"}},
                {"optimizer": optimizer_d,
                 "lr_scheduler": {"scheduler": CosineAnnealingLR(optimizer_d, self.max_epochs, eta_min=self.min_lr),
                                  "interval": "epoch"}})

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.net_D.forward(fake_query)
        else:
            return self.net_D.forward(input_concat)

    def on_train_start(self):
        self.print_to_txt("||start with||\n", self.config)
        self.configure_losses()

    def on_train_epoch_start(self):
        self.print_to_txt("⭐epoch: {}⭐".format(self.current_epoch))
        # 起始时间
        self.train_s_time = time.time()

    def training_step(self, batch, batch_idx):
        real_A, real_B = (batch["image"], batch["t1ce"])
        optimizer_G, optimizer_D = self.optimizers()
        fake_B = self.forward(real_A)
        # =========================D==========================
        self.toggle_optimizer(optimizer_D)
        # fake
        pred_fake = self.discriminate(real_A, fake_B, use_pool=True if self.config.pool_size > 0 else False)
        if "vanilla_GAN_loss" in self.loss_weight_dict.keys():
            loss_D_fake = self.criterion_dict["vanilla_GAN_loss"](pred_fake, False)
        elif "ls_GAN_loss" in self.loss_weight_dict.keys():
            loss_D_fake = self.criterion_dict["ls_GAN_loss"](pred_fake, False)
        # real
        pred_real = self.discriminate(real_A, real_B)
        if "vanilla_GAN_loss" in self.loss_weight_dict.keys():
            loss_D_real = self.criterion_dict["vanilla_GAN_loss"](pred_real, True)
        elif "ls_GAN_loss" in self.loss_weight_dict.keys():
            loss_D_real = self.criterion_dict["ls_GAN_loss"](pred_real, True)

        # loss&backward
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)
        # =========================G==========================
        # fake
        self.toggle_optimizer(optimizer_G)
        pred_fake = self.net_D.forward(input=torch.cat((real_A, fake_B), 1))
        if "vanilla_GAN_loss" in self.loss_weight_dict.keys():
            loss_G_GAN = self.criterion_dict["vanilla_GAN_loss"](pred_fake, True)
        elif "ls_GAN_loss" in self.loss_weight_dict.keys():
            loss_G_GAN = self.criterion_dict["ls_GAN_loss"](pred_fake, True)
        # dist loss
        loss_dist, loss_value_dict = distance_loss(self, fake_B, real_B, pred_fake, pred_real)
        # loss&backward
        loss_G = loss_G_GAN + loss_dist
        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)
        # =========================loss========================================
        loss_dict = {"G_GAN": loss_G_GAN,
                     "fake_D": loss_D_fake,
                     "real_D": loss_D_real}
        loss_dict.update(loss_value_dict)
        if batch_idx % 160 == 0:
            img_realA = torch.cat((real_A[0][0], real_A[0][1]), dim=1).unsqueeze(0).unsqueeze(0)
            img_realA = tensor2im(img_realA)
            img_realB = tensor2im(real_B)
            img_fakeB = tensor2im(fake_B)
            self.logger.experiment.add_image("train_real_A", img_realA, self.global_step, dataformats="HWC")
            self.logger.experiment.add_image("train_real_B", img_realB, self.global_step, dataformats="HWC")
            self.logger.experiment.add_image("train_fake_B", img_fakeB, self.global_step, dataformats="HWC")

        return {"loss_dict": loss_dict}

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        # todo: need to change while loss change
        self.log_dict(outputs["loss_dict"])
        self.train_e_time = time.time()
        time_str = get_duration_time_str(s_time=self.train_s_time, e_time=self.train_e_time)
        loss_print_content = {}
        for loss_n, loss_v in outputs["loss_dict"].items():
            loss_print_content.update({loss_n: "%.4f" % loss_v.item()})
        print_content = "{} / {} {}  || Training cost: {}".format(batch_idx + 1,
                                                                  len(self.train_dataloader()),
                                                                  loss_print_content,
                                                                  time_str)
        printProgressBar(batch_idx, len(self.train_dataloader()) - 1, content=print_content)

    def on_train_epoch_end(self):
        # epoch lr schedule
        sch1, sch2 = self.lr_schedulers()
        if self.current_epoch > self.max_epochs - self.cos_epochs:
            sch1.step()
            sch2.step()

        self.train_e_time = time.time()
        time_str = get_duration_time_str(s_time=self.train_s_time, e_time=self.train_e_time)
        lr = self.optimizers()[0].optimizer.param_groups[0]['lr']
        self.print_to_txt(
            "Epoch Done || lr: {} || Epoch cost: {}".format(lr, time_str))
        self.log("lr", lr)

    def validation_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["t1ce"])
        output = self.forward(images)
        if "Perceptual_loss" in self.loss_weight_dict.keys():
            loss = self.criterion_dict["Perceptual_loss"].forward(output, labels).mean()
        elif "SSIM_loss" in self.loss_weight_dict.keys():
            loss = self.criterion_dict["SSIM_loss"](output, labels)
        elif "VGG_loss" in self.loss_weight_dict.keys():
            fake_B_3C = torch.cat([output] * 3, 1)
            real_B_3C = torch.cat([labels] * 3, 1)
            loss = self.criterion_dict["VGG_loss"](fake_B_3C, real_B_3C)
        else:
            loss = torch.tensor(0)
        # 每15个step保存图片
        # save_image_2d(images, output, labels, self.result_dir, batch_idx, every_n_step=50, mode="Val")
        if batch_idx % 64 == 0:
            img_realB = tensor2im(labels)
            img_fakeB = tensor2im(output)
            self.logger.experiment.add_image("val_real_B", img_realB, self.global_step, dataformats="HWC")
            self.logger.experiment.add_image("val_fake_B", img_fakeB, self.global_step, dataformats="HWC")
        outputs = [i for i in decollate_batch(output)]
        labels = [i for i in decollate_batch(labels)]
        self.mae_metric(y_pred=outputs, y=labels)
        self.ssim_metric(y_pred=outputs, y=labels)
        self.validation_step_outputs.append({"val_loss": loss, "val_number": len(outputs)})

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        printProgressBar(batch_idx, len(self.val_dataloader()) - 1,
                         content="{}/{} validation processing......".format(batch_idx + 1, len(self.val_dataloader())))

    def on_validation_epoch_end(self):
        val_loss, num_items, val_ssim = 0, 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_ssim = self.ssim_metric.aggregate().item()
        mean_val_mae = self.mae_metric.aggregate().item()
        self.mae_metric.reset()
        self.ssim_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        if mean_val_ssim > self.best_val_ssim:
            self.best_val_ssim = mean_val_ssim
            self.best_val_epoch = self.current_epoch
        if mean_val_mae < self.best_val_mae:
            self.best_val_mae = mean_val_mae
        self.print_to_txt(
            f"val_loss: {mean_val_loss:.4f}"
            f" || current mean SSIM: {mean_val_ssim:.4f}"
            f" || best mean SSIM: {self.best_val_ssim:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.print_to_txt(
            f"current mean MAE: {mean_val_mae:.4f}"
            f" || best mean MAE: {self.best_val_mae:.4f} "
        )
        # self.metric_values["MAE"].append(mean_val_mae)
        self.log_dict({
            "val_ssim": mean_val_ssim,
            "val_mae": mean_val_mae,
            "val_loss": mean_val_loss,
        })
        # self.log("best",self.best_val_mae)
        # print_logger(self.metric_values, self.log_pic_dir)
        self.validation_step_outputs.clear()

    def on_predict_start(self):
        self.predict_tic = time.time()
        file_list = sorted(os.listdir(self.test_dir))
        self.pred_dict = dict(zip(file_list, [{} for i in range(len(file_list))]))

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        path, images, labels = (batch["path"], batch["image"], batch["t1ce"])
        id_num = [p.split("/")[-2] for p in path]
        slice_idx = [int(os.path.basename(p).split(".")[0].split("_")[-1]) for p in path]
        roi_x = int(np.ceil(images.shape[2] / 32) * 32)
        roi_y = int(np.ceil(images.shape[3] / 32) * 32)
        roi_size = (roi_x, roi_y)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward, overlap=0.25, mode='gaussian')

        return id_num, outputs, slice_idx

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # 临时文件-2d
        for id_, slice_i, img in zip(outputs[0], outputs[2], outputs[1]):
            output_img = img[0, :, :].cpu().numpy()
            self.pred_dict[id_].update({str(slice_i): output_img})
            # temp_write_path = os.path.join(self.pred_result_dir, "temp", str(id_))
            # if not os.path.exists(temp_write_path):
            #     os.makedirs(temp_write_path)
            # with h5py.File(os.path.join(temp_write_path, "{}.h5".format(slice_i)), "w") as h5file:
            #     h5file["array"] = output_img
            #     h5file.close()
        printProgressBar(batch_idx, len(self.predict_dataloader()) - 1, content="predicting......")

    def on_predict_end(self):
        self.predict_toc = time.time()
        time_str = get_duration_time_str(s_time=self.predict_tic, e_time=self.predict_toc)
        print("predicting cost:", time_str)
        # 处理临时文件
        print("Converting 2d to 3d")
        template_path = os.path.join(self.template_dir, os.path.basename(self.test_dir))
        for idx, id_num in enumerate(self.pred_dict.keys()):
            # all_slice = sorted(os.listdir(os.path.join(temp_dir, id_num)))
            template_nii = sitk.ReadImage(os.path.join(template_path, id_num, "T1CE.nii.gz"))
            template_array = sitk.GetArrayFromImage(template_nii)
            pred_array = np.zeros_like(template_array)
            for slice_idx, slice_img in self.pred_dict[id_num].items():
                pred_array[int(slice_idx)] = slice_img
            pred_nii = sitk.GetImageFromArray(pred_array)
            pred_nii.CopyInformation(template_nii)
            sitk.WriteImage(pred_nii, os.path.join(self.pred_result_dir, "{}_pred.nii.gz".format(id_num)))
            printProgressBar(idx, len(self.pred_dict) - 1,
                             content="{}/{} making prediction nii......".format(idx + 1, len(self.pred_dict)))
        # print("remove temp file......")
        # shutil.rmtree(os.path.join(self.pred_result_dir, "temp"))
        print("Done")
