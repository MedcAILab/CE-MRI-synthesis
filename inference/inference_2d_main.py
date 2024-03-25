import os.path
import re
import sys

import lightning.pytorch as pl
import torch
from lightning.pytorch import seed_everything
from monai.utils import set_determinism

from inference.test_param import config
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD

torch.multiprocessing.set_sharing_strategy('file_system')
set_determinism(2023)
seed_everything(2023, workers=True)

if __name__ == "__main__":
    # torch.set_float32_matmul_precision('high')
    # ==========path============
    Task_name = config.Task_name
    task_id = config.Task_id
    fold_idx = config.fold_idx
    ckpt_name = config.ckpt_name
    dir_prefix = sys.argv[0].split("/newnas")[0]
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # ===============model setting==============
    task_name = "{}_{}_{}_fold5-{}".format(Task_name, task_id, config.net_mode, fold_idx)
    result_path = config.result_path
    # ================search for best==============================
    ckpt_dir = os.path.join(result_path, task_name, "checkpoint")
    ckpt_list = os.listdir(ckpt_dir)
    pattern = r"{}(-epoch=\d+)?\.ckpt".format(ckpt_name.split('.')[0])
    ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
    versions = [re.search(r"epoch=(\d+)", file).group(1) for file in ckpt_file if re.search(r"epoch=\d+", file)]
    sorted_versions = sorted(versions, key=lambda x: int(x))
    ckpt_to_resume = f"{ckpt_name.split('.')[0]}-epoch={sorted_versions[-1]}.ckpt" if sorted_versions else ckpt_name

    ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
    print(ckpt_to_resume)
    model = Pix2Pix_2d_MulD.load_from_checkpoint(ckpt_path,
                                                 map_location={"cuda:4": "cuda:3"}
                                                 )
    model.pred_result_dir = model.pred_result_dir + "_" + ckpt_name.split('.')[0]
    if not os.path.exists(model.pred_result_dir):
        os.makedirs(model.pred_result_dir)
    # ==========PL MODEL============
    torch.set_float32_matmul_precision('high')
    print("========================{}==========================".format(task_name))
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[config.cuda_idx],
        enable_progress_bar=False,
        # limit_predict_batches=2
    )
    predictions = trainer.predict(model)
