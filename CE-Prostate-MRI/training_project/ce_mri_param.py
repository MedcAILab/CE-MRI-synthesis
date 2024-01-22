import argparse

parser = argparse.ArgumentParser()
# =============================常改的参数=============================
parser.add_argument('--net_mode', type=str, default="pix2pix_mulD")
parser.add_argument('--Task_name', type=str, default='CE_MRI_simulate_PCa')  # 任务名,也是文件名
parser.add_argument('--Task_id', type=str, default='1')
parser.add_argument('--cuda_idx', type=int, default=0)  # 用几号卡的显存
parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=1)  # 跑第几折的数据 1开始
parser.add_argument('--vae_local_pretrained', type=bool, default=True)  # 用不用自己本地预训练的VAE

parser.add_argument('--train_keys', type=list, default=["t1", "t2", "b50", "b800", "b1500", "adc"])  # 使用那些序列进行训练
parser.add_argument('--train_batch_size', type=int, default=8)  # 训练的batch_size
parser.add_argument('--num_samples', type=int, default=6)  # 一个病人的batch（最终batch size 等于训练batch size×num_samples）
parser.add_argument('--val_batch_size', type=int, default=10)  # 测试batch_size
parser.add_argument('--val_num_samples', type=int, default=4)  # 测试时一个病人的batch（最终batch size 等于训练batch size×num_samples）

# 判别器
parser.add_argument('--n_layers_D', type=int, default=3)  # 判别器的层数
parser.add_argument('--num_D', type=int, default=2, help='folds number after divided')  # 判别器的个数，下采样次数+1
parser.add_argument('--pool_size', type=int, default=0)  # 图像池的缓存数
# todo 这一句是不能用命令行执行的
parser.add_argument('--loss_weight_dict', type=dict, default={'vanilla_GAN_loss': 1, 'L1_loss': 20, 'SSIM_loss': 20},
                    help="{'vanilla_GAN_loss':1,'L1_loss':50,'SSIM_loss':50,'edge_L1_loss':10}"
                         "{'vanilla_GAN_loss':1,'Perceptual_loss':1}"
                         "'ls_GAN_loss':1,'VGG_loss':10,'G_Feat_loss':10,'MS_SSIM_loss':10")

parser.add_argument('--crop_size', type=tuple, default=(80, 80, 16))
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint_epoch', type=int, default=5)  # 多久保存一个断点
# =============================偶尔改的参数=============================
# dataset_type
parser.add_argument('--dataset_type', type=str, default="normal")  # "normal"
# data-parameters
# parser.add_argument('--filepath_img', type=str, default=r'/data/newnas/MJY_file/CE-MRI/nii_data_norm_pre')
# parser.add_argument('--h5_2d_img_dir', type=str, default=r'/data/newnas/MJY_file/CE-MRI/h5_data_2d_pre')
# parser.add_argument('--filepath_mask', type=str, default=r'/data/newnas/MJY_file/CE-MRI/nii_data_pre')
# # result&save
parser.add_argument('--dir_prefix', type=str, default=r'')
parser.add_argument('--result_path', type=str, default=r'')  # 结果保存地址
parser.add_argument('--filepath_img', type=str,
                    default=r'')
parser.add_argument('--h5_3d_img_dir', type=str, default=r'')
parser.add_argument('--h5_2d_img_dir', type=str, default=r'')
parser.add_argument('--filepath_mask', type=str,
                    default=r'')
# model hyper-parameters
parser.add_argument('--image_size', type=int, default=320)  # no use

parser.add_argument('--lr', type=float, default=5e-5)  # 初始or最大学习率
parser.add_argument('--lr_low', type=float, default=1e-8)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
parser.add_argument('--num_epochs', type=int, default=500)  # 总epoch
parser.add_argument('--lr_cos_epoch', type=int, default=500)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
parser.add_argument('--lr_warm_epoch', type=int, default=0)  # warm_up的epoch数,一般就是10~20,为0或False则不使用

parser.add_argument('--val_step', type=int, default=5)  # 多少epoch测试一次

# =============================一般不改的参数=============================
parser.add_argument('--mode', type=str, default='train', help='train/test')  # 训练还是测试
parser.add_argument('--num_epochs_decay', type=int, default=10)  # decay开始的最小epoch数
parser.add_argument('--decay_ratio', type=float, default=0.1)  # 0~1,每次decay到1*ratio
parser.add_argument('--decay_step', type=int, default=80)  # epoch

# optimizer reg_param
parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--augmentation_prob', type=float, default=0.4)  # 数据扩增的概率

# training hyper-parameters
parser.add_argument('--img_ch', type=int, default=4)
parser.add_argument('--output_ch', type=int, default=1)
parser.add_argument('--DataParallel', type=bool, default=False)  # 数据并行,开了可以用多张卡的显存,不推荐使用
parser.add_argument('--train_flag', type=bool, default=False)  # 训练过程中是否测试训练集,不测试会节省很多时间
parser.add_argument('--seed', type=int, default=2023)  # 随机数的种子点，一般不变
parser.add_argument('--TTA', type=bool, default=False)

config = parser.parse_args()
