import torch

from loss_function.losses_function import SSIM_loss_3d


def distance_loss(self, fake_B, real_B, pred_fake, pred_real):
    loss_value_dict = {}
    loss_dist = 0
    # similarity
    if "L1_loss" in self.loss_weight_dict.keys():
        loss_G_L1 = self.loss_weight_dict["L1_loss"] * self.criterion_dict["L1_loss"](fake_B, real_B)
        loss_value_dict["L1_loss"] = loss_G_L1
        loss_dist += loss_G_L1
    if "SSIM_loss" in self.loss_weight_dict.keys():
        if len(fake_B.shape) == 4:
            loss_G_ssim = self.loss_weight_dict["SSIM_loss"] * self.criterion_dict["SSIM_loss"](fake_B, real_B)
        else:
            loss_G_ssim = self.loss_weight_dict["SSIM_loss"] * SSIM_loss_3d(fake_B, real_B)
        loss_value_dict["SSIM_loss"] = loss_G_ssim
        loss_dist += loss_G_ssim
    if "MS_SSIM_loss" in self.loss_weight_dict.keys():
        loss_G_ssim = -self.loss_weight_dict["MS_SSIM_loss"] * self.criterion_dict["MS_SSIM_loss"](fake_B, real_B)
        loss_value_dict["MS_SSIM_loss"] = loss_G_ssim
        loss_dist += loss_G_ssim
    if "S3IM_loss" in self.loss_weight_dict.keys():
        loss_G_ssim = self.loss_weight_dict["S3IM_loss"] * self.criterion_dict["S3IM_loss"](
            fake_B.view(fake_B.shape[0], -1), real_B.view(fake_B.shape[0], -1),
            ph=fake_B.shape[-2], pw=fake_B.shape[-1])
        loss_value_dict["S3IM_loss"] = loss_G_ssim
        loss_dist += loss_G_ssim
    # perceptual
    if "Perceptual_loss" in self.loss_weight_dict.keys():
        # fake_B_norm = fake_B * 2 - 1
        # real_B_norm = real_B * 2 - 1
        fake_B_n = fake_B
        real_B_n = real_B
        # fake_B_n = (fake_B-fake_B.min())/(fake_B.max()-fake_B.min())
        # real_B_n = (real_B-real_B.min())/(real_B.max()-real_B.min())
        loss_G_perceptual = self.loss_weight_dict["Perceptual_loss"] * self.criterion_dict[
            "Perceptual_loss"].forward(fake_B_n, real_B_n).mean()
        loss_value_dict["Perceptual_loss"] = loss_G_perceptual
        loss_dist += loss_G_perceptual
    if "VGG_loss" in self.loss_weight_dict.keys():
        fake_B_3C = torch.cat([fake_B] * 3, 1)
        real_B_3C = torch.cat([real_B] * 3, 1)
        loss_G_VGG = self.criterion_dict["VGG_loss"](fake_B_3C, real_B_3C) * self.loss_weight_dict["VGG_loss"]  # 10
        loss_value_dict["VGG_loss"] = loss_G_VGG
        loss_dist += loss_G_VGG
    if "G_Feat_loss" in self.loss_weight_dict.keys():
        loss_G_Feat = 0
        feat_weights = 4.0 / (self.config.n_layers_D + 1)
        D_weights = 1.0 / self.config.num_D
        for i in range(self.config.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_Feat += D_weights * feat_weights * \
                               self.criterion_dict["G_Feat_loss"](pred_fake[i][j], pred_real[i][j].detach()) * \
                               self.loss_weight_dict["G_Feat_loss"]  # 10
        loss_value_dict["G_Feat_loss"] = loss_G_Feat
        loss_dist += loss_G_Feat
    return loss_dist, loss_value_dict
