from net.pix2pix_HD_model.networks import define_D
from segmentation_models_pytorch import Unet

net_G = Unet(encoder_name='timm-regnety_160',
             encoder_weights='imagenet',
             encoder_depth=4,
             decoder_channels=[128, 64, 32, 16],
             decoder_use_batchnorm=True,
             in_channels=6, classes=1).to('cuda')
net_D = define_D(input_nc=7,
                 ndf=64,
                 n_layers_D=3,
                 norm="instance",
                 use_sigmoid=False,
                 num_D=2,
                 getIntermFeat=False)
print(net_D)
