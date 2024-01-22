import ants
import lpips
import numpy as np
import torch
import torchmetrics
from nipype.interfaces.ants import MeasureImageSimilarity
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as struct_sim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import normalized_mutual_info_score as sk_mi

from loss_function.perceptual_loss import PerceptualLoss

lpips_loss_fn = lpips.LPIPS(net='vgg', spatial=False, lpips=False)

fid = torchmetrics.image.fid.FrechetInceptionDistance(normalize=False)


# scale to 12 bit range
def scale12bit(img):
    # constants
    new_mean = 2048.
    new_std = 400.

    return np.clip(((img - np.mean(img)) / (np.std(img) / new_std)) + new_mean, 1e-10, 4095)


def scale256(img):
    # constants
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img.astype(np.uint8)


# get list of locals
start_globals = list(globals().keys())


# perform image comparrison using neighborhood CC
def cc_py(true_path, pred_path, mask):
    gt = ants.image_read(true_path)
    pred = ants.image_read(pred_path)
    mask = ants.image_read(mask)
    cc_metric = ants.image_similarity(fixed_image=gt,
                                      moving_image=pred,
                                      fixed_mask=mask,
                                      moving_mask=mask,
                                      metric_type="ANTSNeighborhoodCorrelation",
                                      sampling_strategy='None',
                                      # sampling_percentage=1.0
                                      )
    return -cc_metric


def cc(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    sim = MeasureImageSimilarity()
    if not verbose:
        sim.terminal_output = 'allatonce'
    sim.inputs.dimension = 3
    sim.inputs.metric = 'CC'
    sim.inputs.fixed_image = true_nii
    sim.inputs.moving_image = pred_nii
    sim.inputs.metric_weight = 1.0
    sim.inputs.radius_or_number_of_bins = 5
    sim.inputs.sampling_strategy = 'None'  # None = dense sampling
    sim.inputs.sampling_percentage = 1.0
    if mask:
        sim.inputs.fixed_image_mask = mask_nii
        sim.inputs.moving_image_mask = mask_nii
    if verbose:
        print(sim.cmdline)

    return np.abs(sim.run().outputs.similarity)


# perform image comparrison using histogram MI
def mi_py(true_path, pred_path, mask):
    gt = ants.image_read(true_path)
    pred = ants.image_read(pred_path)
    mask = ants.image_read(mask)
    mi_metric = ants.image_similarity(fixed_image=gt,
                                      moving_image=pred,
                                      fixed_mask=mask,
                                      moving_mask=mask,
                                      metric_type="MattesMutualInformation",
                                      sampling_strategy='None',
                                      # sampling_percentage=1.0
                                      )
    return -mi_metric


def nmi(true_array, pred_array, mask=None):
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(true_array, dtype=bool)
    # scale to 12 bit range
    true_img = scale256(true_array[mask_img])
    pred_img = scale256(pred_array[mask_img])

    return sk_mi(true_img, pred_img)


def mi(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    sim = MeasureImageSimilarity()
    if not verbose:
        sim.terminal_output = 'allatonce'
    sim.inputs.dimension = 3
    sim.inputs.metric = 'MI'
    sim.inputs.fixed_image = true_nii
    sim.inputs.moving_image = pred_nii
    sim.inputs.metric_weight = 1.0
    sim.inputs.radius_or_number_of_bins = 64
    sim.inputs.sampling_strategy = 'None'  # None = dense sampling
    sim.inputs.sampling_percentage = 1.0
    if mask:
        sim.inputs.fixed_image_mask = mask_nii
        sim.inputs.moving_image_mask = mask_nii
    if verbose:
        print(sim.cmdline)

    return np.abs(sim.run().outputs.similarity)


# perform image comparrison using histogram MI
def mse(true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    sim = MeasureImageSimilarity()
    if not verbose:
        sim.terminal_output = 'allatonce'
    sim.inputs.dimension = 3
    sim.inputs.metric = 'MeanSquares'
    sim.inputs.fixed_image = true_nii
    sim.inputs.moving_image = pred_nii
    sim.inputs.metric_weight = 1.0
    sim.inputs.radius_or_number_of_bins = 32  # not used
    sim.inputs.sampling_strategy = 'None'  # None = dense sampling
    sim.inputs.sampling_percentage = 1.0
    if mask:
        sim.inputs.fixed_image_mask = mask_nii
        sim.inputs.moving_image_mask = mask_nii
    if verbose:
        print(sim.cmdline)

    return np.abs(sim.run().outputs.similarity)


# normalized RMS error
def nrmse(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(true_array, dtype=bool)
    # verbosity
    if verbose:
        print("Calculating nrmse")

    return mean_squared_error(true_array[mask_img], pred_array[mask_img], squared=False) / (
            np.max(true_array[mask_img]) - np.min(true_array[mask_img]))


# mean absolute percentage error
def mape(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(true_array, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    # verbosity
    if verbose:
        print("Calculating MAPE on 12 bit range scaled images")

    return np.mean(np.fabs((true_img - pred_img)) / (np.fabs(true_img)))


# symmetric mean absolute percentage error
def smape(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(true_array, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    # verbosity
    if verbose:
        print("Calculating sMAPE on 12 bit range scaled images")

    return np.mean(np.fabs(pred_img - true_img) / (np.fabs(true_img) + np.fabs(pred_img)))


# log of accuracy ratio
def logac(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(true_array, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    # verbosity
    if verbose:
        print("Calculating logac on 12 bit range scaled images")

    return np.mean(np.fabs((np.log(pred_img / true_img))))


# median symmetric accuracy (cf. Morley, 2016)
def medsymac(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(true_array, dtype=bool)
    # scale to 12 bit range
    true_img = scale12bit(true_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    # verbosity
    if verbose:
        print("Calculating medsymac on 12 bit range scaled images")

    return np.exp(np.median(np.fabs(np.log(pred_img / true_img)))) - 1


# structural similarity index
def ssim(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = mask.astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        true_img = scale12bit(true_img)
        pred_img = scale12bit(pred_img)
        # ssim_final = 0
        # for i in range(true_img.shape[0]):
        #     true_img_slice = true_img[i, ...]
        #     pred_img_slice = pred_img[i, ...]
        #     ssim_final += struct_sim(true_img_slice, pred_img_slice, win_size=9, data_range=true_img.max() - true_img.min())
    # verbosity
    if verbose:
        print("Calculating structural similarity index")
    # data_range = true_img.max() - true_img.min()
    # print(data_range)
    return struct_sim(true_img, pred_img, win_size=9, data_range=true_img.max() - true_img.min())


def ssim_torch(true_array, pred_array, mask=None, verbose=False):
    ms_ssim = torchmetrics.functional.image.multiscale_structural_similarity_index_measure
    if mask is not None:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = mask.astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        true_img = scale12bit(true_img)
        pred_img = scale12bit(pred_img)
        true_tensor = torch.from_numpy(true_img).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0)
        ms_ssim_final = 0
        for i in range(true_img.shape[0]):
            true_tensor_slice = true_tensor[:, :, i, ...]
            pred_tensor_slice = pred_tensor[:, :, i, ...]
            ms_ssim_final += ms_ssim(preds=true_tensor_slice, target=pred_tensor_slice)
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return (ms_ssim_final / true_img.shape[0]).item()


def ssim_torch_s(true_array, pred_array, mask=None, verbose=False):
    ssim = torchmetrics.functional.image.structural_similarity_index_measure
    if mask is not None:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = mask.astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        true_img = scale12bit(true_img)
        pred_img = scale12bit(pred_img)
        true_tensor = torch.from_numpy(true_img).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0)
        ssim_final = 0
        for i in range(true_img.shape[0]):
            true_tensor_slice = true_tensor[:, :, i, ...]
            pred_tensor_slice = pred_tensor[:, :, i, ...]
            ssim_final += ssim(preds=true_tensor_slice, target=pred_tensor_slice)

        # true_img = scale12bit(true_img)
        # pred_img = scale12bit(pred_img)
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return (ssim_final / true_img.shape[0]).item()


def fid_torch(true_array, pred_array, mask=None, verbose=False, compute=False):
    if mask is not None:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = mask.astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        true_img = scale256(true_img)
        pred_img = scale256(pred_img)
        true_tensor = torch.from_numpy(true_img).unsqueeze(1)
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(1)
        fid_final = 0
        true_tensor_slice = torch.cat([true_tensor, true_tensor, true_tensor], dim=1)
        pred_tensor_slice = torch.cat([pred_tensor, pred_tensor, pred_tensor], dim=1)
        fid.update(true_tensor_slice, real=True)
        fid.update(pred_tensor_slice, real=False)
        if compute:
            fid_final = fid.compute().item()
        else:
            fid_final = 0
        # fid.reset()
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return fid_final


def vif_torch(true_array, pred_array, mask=None, verbose=False):
    vif = torchmetrics.image.VisualInformationFidelity()
    if mask is not None:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = mask.astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        # true_img = scale256(true_img)
        # pred_img = scale256(pred_img)
        true_tensor = torch.from_numpy(true_img).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0)
        fid_final = 0
        for i in range(true_img.shape[0]):
            true_tensor_slice = true_tensor[:, :, i, ...]
            pred_tensor_slice = pred_tensor[:, :, i, ...]
            fid_final += vif(pred_tensor_slice, true_tensor_slice)
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return (fid_final / true_img.shape[0]).item()


def psnr(true_array, pred_array, mask=None, verbose=False):
    if mask is not None:  # crop image to mask to avoid black space counting as similarity - cant vectorize this operation?
        # tight cropping to mask should already be done as part of evaluate.py, but including here for compatibility
        mask_img = mask.astype(bool)
        nzi = np.nonzero(mask_img)
        true_img = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_img = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        true_img = scale12bit(true_img)
        pred_img = scale12bit(pred_img)
    # verbosity
    if verbose:
        print("Calculating structural similarity index")

    return compute_psnr(true_img, pred_img, data_range=true_img.max() - true_img.min())


def metric_picker(metric, true_nii, pred_nii, mask_nii, mask=False, verbose=False):
    # sanity checks
    if not isinstance(metric, str):
        raise ValueError("Metric parameter must be a string")

    # check for specified loss method and error if not found
    if metric.lower() in globals():
        metric_val = globals()[metric.lower()](true_nii, pred_nii, mask_nii, mask, verbose)
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified metric: '{}' is not one of the available methods: {}".format(metric, methods))

    return metric_val


def lpips_metric(true_array, pred_array, mask=None):
    loss_fn = lpips_loss_fn
    dist = 0
    if mask is not None:
        nzi = np.nonzero(mask)
        true_array = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_array = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        true_array = scale256(true_array)
        pred_array = scale256(pred_array)
    for layer in range(true_array.shape[0]):
        true_tensor = torch.from_numpy(
            np.tile(true_array[layer:layer + 1, :, :, np.newaxis].transpose(3, 0, 1, 2), (1, 3, 1, 1)))
        pred_tensor = torch.from_numpy(
            np.tile(pred_array[layer:layer + 1, :, :, np.newaxis].transpose(3, 0, 1, 2), (1, 3, 1, 1)))
        # pred_tensor = pred_tensor.cuda()
        # true_tensor = true_tensor.cuda()
        dist += loss_fn.forward(true_tensor, pred_tensor, normalize=True)
    dist = dist / true_array.shape[0]
    return dist.item()


def med_lpips_metric(true_array, pred_array, mask=None):
    Perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50", )
    dist = 0
    if mask is not None:
        nzi = np.nonzero(mask)
        true_array = true_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
        pred_array = pred_array[nzi[0].min():nzi[0].max(), nzi[1].min():nzi[1].max(), nzi[2].min():nzi[2].max()]
    for layer in range(true_array.shape[0]):
        true_tensor = torch.from_numpy(true_array[layer:layer + 1, :, :, np.newaxis].transpose(3, 0, 1, 2))
        pred_tensor = torch.from_numpy(pred_array[layer:layer + 1, :, :, np.newaxis].transpose(3, 0, 1, 2))
        # pred_tensor = pred_tensor.cuda()
        # true_tensor = true_tensor.cuda()
        dist += Perceptual_loss.forward(pred_tensor, true_tensor)
    dist = dist / true_array.shape[0]
    return dist.item()
