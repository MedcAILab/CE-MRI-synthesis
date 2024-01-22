import argparse

parser = argparse.ArgumentParser()
# =============================常改的参数=============================
parser.add_argument('--Task_name', type=str, default='CE_MRI_simulate_PCa')  # 任务名,也是文件名
parser.add_argument('--Task_id', type=str, default='1')
parser.add_argument('--cuda_idx', type=int, default=0)  # 用几号卡的显存
parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=1)  # 跑第几折的数据 1开始
parser.add_argument('--ckpt_name', type=str, default="checkpoint.ckpt", help="best/checkpoint")
parser.add_argument('--net_mode', type=str, default="pix2pix_mulD")
# image
# parser.add_argument('--filepath_img', type=str, default=r'/data/newnas/MJY_file/CE-MRI/nii_data_norm_pre')
parser.add_argument('--filepath_img', type=str, default=r'')
# result&save
parser.add_argument('--result_path', type=str, default=r'')
# training hyper-parameters
parser.add_argument('--seed', type=int, default=2023)  # 随机数的种子点，一般不变
config = parser.parse_args()
