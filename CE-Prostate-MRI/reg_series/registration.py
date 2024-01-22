import json
# import pydicom as dicom
import logging
import multiprocessing
import os

from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.ants import Registration


def get_nii_file_name(sdict, workdir):
    series_file_list = os.listdir(workdir)

    for series in sorted(sdict.keys()):
        if "filename" not in sdict[series].keys():
            # 遍历json的可能的名字序列
            for series_dict_name in sdict[series]["name"]:
                # 找对应的实际文件名
                for series_file_name in series_file_list:
                    # json名字对应上实际文件名
                    if series_dict_name in series_file_name:
                        sdict[series].update({"filename": os.path.join(workdir, series_file_name)})
                        break

    return sdict


def get_series(workdir, params):
    assert os.path.isfile(params), "Param file does not exist at " + params
    with open(params, 'r') as f:
        json_str = f.read()
    sdict = json.loads(json_str)
    sdict = get_nii_file_name(sdict, workdir)
    sdict.update({"info": {
        "filename": "None",
        "dcmdir": workdir,
        "id": workdir.split(os.sep)[-1],
    }})
    return sdict


def make_log(work_dir, repeat=False):
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    # make log file, append to existing
    idno = os.path.basename(work_dir)
    log_file = os.path.join(work_dir, idno + "_log.txt")
    if repeat:
        open(log_file, 'w').close()
    else:
        open(log_file, 'a').close()
    # make logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # should this be DEBUG?
    # set all existing handlers to null to prevent duplication
    logger.handlers = []
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterch = logging.Formatter('%(message)s')
    formatterfh = logging.Formatter("[%(asctime)s]  [%(levelname)s]:     %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatterch)
    fh.setFormatter(formatterfh)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("####################### STARTING NEW LOG #######################")


# Fast ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def affine_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args = '--float'
    antsreg.inputs.fixed_image = template_nii
    antsreg.inputs.moving_image = moving_nii
    antsreg.inputs.output_transform_prefix = outprefix
    antsreg.inputs.num_threads = multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas = [[6, 4, 1, 0], [6, 4, 1, 0]]
    antsreg.inputs.sigma_units = ['mm', 'mm']
    antsreg.inputs.transforms = ['Rigid', 'Affine']
    antsreg.terminal_output = 'none'
    antsreg.inputs.use_histogram_matching = True
    antsreg.inputs.write_composite_transform = True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.metric = ['Mattes', 'Mattes']
    antsreg.inputs.metric_weight = [1.0, 1.0]
    antsreg.inputs.number_of_iterations = [[1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000]]
    antsreg.inputs.convergence_threshold = [1e-07, 1e-07]
    antsreg.inputs.convergence_window_size = [10, 10]
    antsreg.inputs.radius_or_number_of_bins = [32, 32]
    antsreg.inputs.sampling_strategy = ['Regular', 'Regular']
    antsreg.inputs.sampling_percentage = [0.25, 0.25]  # 1
    antsreg.inputs.shrink_factors = [[4, 3, 2, 1], [4, 3, 2, 1]]
    antsreg.inputs.transform_parameters = [(0.1,), (0.1,)]

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Faster ants affine
# takes moving and template niis and a work dir
# performs fast affine registration and returns a list of transforms
def fast_affine_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args = '--float'
    antsreg.inputs.fixed_image = template_nii
    antsreg.inputs.moving_image = moving_nii
    antsreg.inputs.output_transform_prefix = outprefix
    antsreg.inputs.num_threads = multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas = [[6, 4, 1], [6, 4, 1]]
    antsreg.inputs.sigma_units = ['mm', 'mm']
    antsreg.inputs.transforms = ['Rigid', 'Affine']
    antsreg.terminal_output = 'none'
    antsreg.inputs.use_histogram_matching = True
    antsreg.inputs.write_composite_transform = True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.metric = ['Mattes', 'Mattes']
    antsreg.inputs.metric_weight = [1.0, 1.0]
    antsreg.inputs.number_of_iterations = [[1000, 1000, 1000], [1000, 1000, 1000]]
    antsreg.inputs.convergence_threshold = [1e-04, 1e-04]
    antsreg.inputs.convergence_window_size = [5, 5]
    antsreg.inputs.radius_or_number_of_bins = [32, 32]
    antsreg.inputs.sampling_strategy = ['Regular', 'Regular']
    antsreg.inputs.sampling_percentage = [0.25, 0.25]
    antsreg.inputs.shrink_factors = [[6, 4, 2], [6, 4, 2]] * 2
    antsreg.inputs.transform_parameters = [(0.1,), (0.1,)]

    trnsfm = outprefix + "Composite.h5"
    # h5 = h5py.File(trnsfm,'w')
    # h5.close()

    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        workingdir = os.path.dirname(outprefix)
        print('True' if os.path.isdir(workingdir) else 'False')
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Fast ants diffeomorphic registration
# takes moving and template niis and a work dir
# performs fast diffeomorphic registration and returns a list of transforms
def diffeo_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args = '--float'
    antsreg.inputs.fixed_image = template_nii
    antsreg.inputs.moving_image = moving_nii
    antsreg.inputs.output_transform_prefix = outprefix
    antsreg.inputs.num_threads = multiprocessing.cpu_count()
    antsreg.terminal_output = 'none'
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.shrink_factors = [[4, 3, 2, 1], [8, 4, 2, 1], [4, 2, 1]]
    antsreg.inputs.smoothing_sigmas = [[6, 4, 1, 0], [4, 2, 1, 0], [2, 1, 0]]
    antsreg.inputs.sigma_units = ['mm', 'mm', 'mm']
    antsreg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    antsreg.inputs.use_histogram_matching = [True, True, True]
    antsreg.inputs.write_composite_transform = True
    antsreg.inputs.metric = ['Mattes', 'Mattes', 'Mattes']
    antsreg.inputs.metric_weight = [1.0, 1.0, 1.0]
    antsreg.inputs.number_of_iterations = [[1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000], [250, 100, 50]]
    antsreg.inputs.convergence_threshold = [1e-07, 1e-07, 1e-07]
    antsreg.inputs.convergence_window_size = [5, 5, 5]
    antsreg.inputs.radius_or_number_of_bins = [32, 32, 32]
    antsreg.inputs.sampling_strategy = ['Regular', 'Regular', 'None']  # 'None'
    antsreg.inputs.sampling_percentage = [0.25, 0.25, 1]
    antsreg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Faster ants diffeomorphic registration
# takes moving and template niis and a work dir
# performs fast diffeomorphic registration and returns a list of transforms
def fast_diffeo_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args = '--float'
    antsreg.inputs.fixed_image = template_nii
    antsreg.inputs.moving_image = moving_nii
    antsreg.inputs.output_transform_prefix = outprefix
    antsreg.inputs.num_threads = multiprocessing.cpu_count()
    antsreg.terminal_output = 'none'
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.shrink_factors = [[6, 4, 2], [4, 2]]
    antsreg.inputs.smoothing_sigmas = [[4, 2, 1], [2, 1]]
    antsreg.inputs.sigma_units = ['mm', 'mm']
    antsreg.inputs.transforms = ['Affine', 'SyN']
    antsreg.inputs.use_histogram_matching = [True, True]
    antsreg.inputs.write_composite_transform = True
    antsreg.inputs.metric = ['Mattes', 'Mattes']
    antsreg.inputs.metric_weight = [1.0, 1.0]
    antsreg.inputs.number_of_iterations = [[1000, 500, 250], [50, 50]]
    antsreg.inputs.convergence_threshold = [1e-05, 1e-05]
    antsreg.inputs.convergence_window_size = [5, 5]
    antsreg.inputs.radius_or_number_of_bins = [32, 32]
    antsreg.inputs.sampling_strategy = ['Regular', 'None']  # 'None'
    antsreg.inputs.sampling_percentage = [0.25, 1]
    antsreg.inputs.transform_parameters = [(0.1,), (0.1, 3.0, 0.0)]

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# ANTS translation
# takes moving and template niis and a work dir
# performs fast translation only registration and returns a list of transforms
def trans_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args = '--float'
    antsreg.inputs.fixed_image = template_nii
    antsreg.inputs.moving_image = moving_nii
    antsreg.inputs.output_transform_prefix = outprefix
    antsreg.inputs.num_threads = multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas = [[6, 4, 1, 0]]
    antsreg.inputs.sigma_units = ['vox']
    antsreg.inputs.transforms = ['Translation']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output = 'none'
    antsreg.inputs.use_histogram_matching = True
    antsreg.inputs.write_composite_transform = True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.metric = ['Mattes']  # ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight = [1.0]
    antsreg.inputs.number_of_iterations = [[1000, 500, 250, 50]]  # [100, 70, 50, 20]
    antsreg.inputs.convergence_threshold = [1e-07]
    antsreg.inputs.convergence_window_size = [10]
    antsreg.inputs.radius_or_number_of_bins = [32]  # 4
    antsreg.inputs.sampling_strategy = ['Regular']  # 'None'
    antsreg.inputs.sampling_percentage = [0.25]  # 1
    antsreg.inputs.shrink_factors = [[4, 3, 2, 1]]  # *3
    antsreg.inputs.transform_parameters = [(0.1,)]  # (0.1, 3.0, 0.0) # affine gradient step

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# ANTS translation
# takes moving and template niis and a work dir
# performs fast translation only registration and returns a list of transforms
def rigid_reg(moving_nii, template_nii, work_dir, option=None, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # get basenames
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    outprefix = os.path.join(work_dir, moving_name + "_2_" + template_name + "_")

    # registration setup
    antsreg = Registration()
    antsreg.inputs.args = '--float'
    antsreg.inputs.fixed_image = template_nii
    antsreg.inputs.moving_image = moving_nii
    antsreg.inputs.output_transform_prefix = outprefix
    antsreg.inputs.num_threads = multiprocessing.cpu_count()
    antsreg.inputs.smoothing_sigmas = [[6, 4, 1, 0]]
    antsreg.inputs.sigma_units = ['vox']
    antsreg.inputs.transforms = ['Rigid']  # ['Rigid', 'Affine', 'SyN']
    antsreg.terminal_output = 'none'
    antsreg.inputs.use_histogram_matching = True
    antsreg.inputs.write_composite_transform = True
    if isinstance(option, dict) and "reg_com" in option.keys():
        antsreg.inputs.initial_moving_transform_com = option["reg_com"]
    else:
        antsreg.inputs.initial_moving_transform_com = 1  # use center of mass for initial transform by default
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.metric = ['Mattes']  # ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight = [1.0]
    antsreg.inputs.number_of_iterations = [[1000, 500, 250, 50]]  # [100, 70, 50, 20]
    antsreg.inputs.convergence_threshold = [1e-07]
    antsreg.inputs.convergence_window_size = [10]
    antsreg.inputs.radius_or_number_of_bins = [32]  # 4
    antsreg.inputs.sampling_strategy = ['Regular']  # 'None'
    antsreg.inputs.sampling_percentage = [0.25]  # 1
    antsreg.inputs.shrink_factors = [[4, 3, 2, 1]]  # *3
    antsreg.inputs.transform_parameters = [(0.1,)]  # (0.1, 3.0, 0.0) # affine gradient step

    trnsfm = outprefix + "Composite.h5"
    if not os.path.isfile(trnsfm) or repeat:
        logger.info("- Registering image " + moving_nii + " to " + template_nii)
        logger.debug(antsreg.cmdline)
        antsreg.run()
    else:
        logger.info("- Warp file already exists at " + trnsfm)
        logger.debug(antsreg.cmdline)
    return trnsfm


# Ants apply transforms to list
# takes moving and reference niis, an output filename, plus a transform list
# applys transform and saves output as output_nii
def ants_apply(moving_nii, reference_nii, interp, transform_list, work_dir, invert_bool=False, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    # enforce list
    if not isinstance(moving_nii, list):
        moving_nii = [moving_nii]
    if not isinstance(transform_list, list):
        transform_list = [transform_list]
    # create output list of same shape
    output_nii = moving_nii
    # define extension
    ext = ".nii"
    # for loop for applying reg
    for ind, mvng in enumerate(moving_nii, 0):
        # define output path
        output_nii[ind] = os.path.join(work_dir, os.path.basename(mvng).split(ext)[0] + '_w.nii.gz')
        # do registration if not already done
        antsapply = ApplyTransforms()
        antsapply.inputs.dimension = 3
        antsapply.terminal_output = 'none'  # suppress terminal output
        antsapply.inputs.input_image = mvng
        antsapply.inputs.reference_image = reference_nii
        antsapply.inputs.output_image = output_nii[ind]
        antsapply.inputs.interpolation = interp
        antsapply.inputs.default_value = 0
        antsapply.inputs.transforms = transform_list
        antsapply.inputs.invert_transform_flags = [invert_bool] * len(transform_list)
        if not os.path.isfile(output_nii[ind]) or repeat:
            logger.info("- Creating warped image " + output_nii[ind])
            logger.debug(antsapply.cmdline)
            antsapply.run()
        else:
            logger.info("- Transformed image already exists at " + output_nii[ind])
            logger.debug(antsapply.cmdline)
    # if only 1 label, don't return array
    if len(output_nii) == 1:
        output_nii = output_nii[0]
    return output_nii


# register data together using reg_target as target, if its a file, use it
# if not assume its a dict key for an already registered file
# there are multiple loops here because dicts dont preserve order, and we need it for some registration steps
def reg_series(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("REGISTERING IMAGES:")
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    # sort serdict keys so that the atlas reg comes up first - this makes sure atlas registration is first
    sorted_keys = []
    for key in sorted(ser_dict.keys()):
        if "reg_target" in ser_dict[key] and ser_dict[key]["reg_target"] == "atlas":
            sorted_keys.insert(0, key)
        else:
            sorted_keys.append(key)
    # make sure serdict keys requiring other registrations to be performed first are at end of list
    last = []
    tmp = []
    for key in sorted_keys:
        if "reg_last" in ser_dict[key] and ser_dict[key]["reg_last"]:
            last.append(key)
        else:
            tmp.append(key)
    for key in last:
        tmp.append(key)
    sorted_keys = tmp

    # if reg is false, or if there is no input file found, then just make the reg filename same as unreg filename
    for ser in sorted_keys:
        # first, if there is no filename, set to None
        if "filename" not in ser_dict[ser].keys():
            ser_dict[ser].update({"filename": "None"})
        if ser_dict[ser]["filename"] == "None" or "reg" not in ser_dict[ser].keys() or not ser_dict[ser]["reg"]:
            ser_dict[ser].update({"filename_reg": ser_dict[ser]["filename"]})
            ser_dict[ser].update({"transform": "None"})
            ser_dict[ser].update({"reg": False})
        # if reg True, then do the registration using translation, affine, nonlin, or just applying existing transform
    # handle translation registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] == "trans":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            # handle registration options here
            if "reg_option" in ser_dict[ser].keys():
                option = ser_dict[ser]["reg_option"]
            else:
                option = None
            transforms = trans_reg(movingr, template, dcm_dir, option, repeat)
            # handle interp option
            if "interp" in ser_dict[ser].keys():
                interp = ser_dict[ser]["interp"]
            else:
                interp = 'Linear'
            niiout = ants_apply(movinga, template, interp, transforms, dcm_dir, repeat)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms})
    # handle rigid registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] == "rigid":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            # handle registration options here
            if "reg_option" in ser_dict[ser].keys():
                option = ser_dict[ser]["reg_option"]
            else:
                option = None
            transforms = rigid_reg(movingr, template, dcm_dir, option, repeat)
            # handle interp option
            if "interp" in ser_dict[ser].keys():
                interp = ser_dict[ser]["interp"]
            else:
                interp = 'Linear'
            niiout = ants_apply(movinga, template, interp, transforms, dcm_dir, repeat)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms})
    # handle affine registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] == "affine":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template):  # make sure template and moving files exist
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = affine_reg(movingr, template, dcm_dir, option, repeat)
                # handle interp option
                if "interp" in ser_dict[ser].keys():
                    interp = ser_dict[ser]["interp"]
                else:
                    interp = 'Linear'
                niiout = ants_apply(movinga, template, interp, transforms, dcm_dir, repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle faster affine registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] == "fast_affine":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template):  # make sure template and moving files exist
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = fast_affine_reg(movingr, template, dcm_dir, option, repeat)
                # handle interp option
                if "interp" in ser_dict[ser].keys():
                    interp = ser_dict[ser]["interp"]
                else:
                    interp = 'Linear'
                niiout = ants_apply(movinga, template, interp, transforms, dcm_dir, repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle diffeo registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] == "diffeo":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template):  # check that all files exist prior to reg
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = diffeo_reg(movingr, template, dcm_dir, option, repeat)
                # handle interp option
                if "interp" in ser_dict[ser].keys():
                    interp = ser_dict[ser]["interp"]
                else:
                    interp = 'Linear'
                niiout = ants_apply(movinga, template, interp, transforms, dcm_dir, repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle faster diffeo registration
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] == "fast_diffeo":
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            if os.path.isfile(movingr) and os.path.isfile(template):  # check that all files exist prior to reg
                # handle registration options here
                if "reg_option" in ser_dict[ser].keys():
                    option = ser_dict[ser]["reg_option"]
                else:
                    option = None
                transforms = fast_diffeo_reg(movingr, template, dcm_dir, option, repeat)
                # handle interp option
                if "interp" in ser_dict[ser].keys():
                    interp = ser_dict[ser]["interp"]
                else:
                    interp = 'Linear'
                niiout = ants_apply(movinga, template, interp, transforms, dcm_dir, repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
    # handle applying an existing transform (assumes reg entry is the key for another series' transform)
    for ser in sorted_keys:
        if ser_dict[ser]["reg"] in sorted_keys:
            try:
                transforms = ser_dict[ser_dict[ser]["reg"]]["transform"]
                template = ser_dict[ser_dict[ser]["reg"]]["filename_reg"]
                moving = ser_dict[ser]["filename"]
                # handle interp option
                if "interp" in ser_dict[ser].keys():
                    interp = ser_dict[ser]["interp"]
                else:
                    interp = 'Linear'
                niiout = ants_apply(moving, template, interp, transforms, dcm_dir, repeat)
                ser_dict[ser].update({"filename_reg": niiout})
                ser_dict[ser].update({"transform": transforms})
            except Exception:
                logger.info("- Error attempting to apply existing transform to seies {}".format(ser))
    return ser_dict
