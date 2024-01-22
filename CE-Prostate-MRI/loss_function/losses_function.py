import torch
from monai.losses import SSIMLoss
from net.pix2pix_HD_model import networks
from torch.nn import MSELoss, L1Loss
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

from loss_function.perceptual_loss import PerceptualLoss
from loss_function.s3im.s3im import S3IM
from training_project.ce_mri_param import config

device = torch.device("cuda:{}".format(config.cuda_idx))
start_globals = list(globals().keys())

# ===========losses================
MSE_loss = MSELoss()
edge_L1_loss = L1_loss = L1Loss()
vanilla_GAN_loss = networks.GANLoss(use_lsgan=False)
ls_GAN_loss = networks.GANLoss(use_lsgan=True)
SSIM_loss = SSIMLoss(spatial_dims=2)
SSIM_loss_3d = SSIMLoss(spatial_dims=3)
S3IM_loss = S3IM(kernel_size=7)
# MS_SSIM_loss = MultiScaleStructuralSimilarityIndexMeasure().to(device)
MS_SSIM_loss = multiscale_structural_similarity_index_measure
if "Perceptual_loss" in config.loss_weight_dict.keys():
    Perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50", ).cuda(int(config.cuda_idx))
if "VGG_loss" in config.loss_weight_dict.keys():
    VGG_loss = networks.VGGLoss(int(config.cuda_idx))
if "G_Feat_loss" in config.loss_weight_dict.keys():
    G_Feat_loss = L1Loss()


# =================================

def loss_picker(loss_name):
    if not isinstance(loss_name, str):
        raise ValueError("Loss method parameter must be a string")
    # check for specified loss method and error if not found
    if loss_name in globals():
        loss_fn = globals()[loss_name]
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified loss method: '{}' is not one of the available methods: {}".format(loss_name, methods[1:-1]))

    return loss_fn


if __name__ == "__main__":
    a = loss_picker("SSIM_loss")
    b = SSIMLoss(spatial_dims=2)
    print(a)
    print(b)
