# CE-MRI-synthesis
This project utilizes cGAN with a RegNet backbone to generate contrast-enhanced MRI from a set of pre-contrast MRI sequences  

The code is based on Pytorch lightning

## 1. Data preprocess
**Follow the workflow outlined below.**

First use the `get_body_t1.py` to generate the mask of T1 or anything you want.  

Than use `get_body_all_series.py` to apply the mask on every sequence.  

Use `reg_series/resample_t1.py` to resample every T1 to a chosen T1. (ensuring consistent size and spacing)

`reg_series/itk_resample_all2t1.py` to resample every sequence to T1 within a patient

`reg_series/kill_empty_slice.py` to kill the empty slice if needed

`reg_series/preprocess.py` to register the sequences of each patient to ensure organ and body alignment.
## 2. Run train  
Edit the `training_project/ce_mri_param.py` to change the configuration

Than you can run the `train_main.py`

## 3. Inference and test

Edit the `inference/test_param.py` and run `inference/inference_2d_main.py`
