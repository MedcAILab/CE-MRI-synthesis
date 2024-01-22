"""
自定义transform,在这个文件自行修改
"""
import math

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    ToTensord,
    LoadImaged,
    RandFlipd,
    RandCropByPosNegLabeld,
    ConcatItemsd,
    RandRotated,
    DivisiblePadd,
    RepeatChanneld
)

from training_project.utils.my_transform import LoadH5, LoadImageITKd


def get_train_transform(keys, crop_size, random_prob, num_samples):
    train_transforms = Compose(
        [
            # 固定的transform
            LoadImaged(keys=keys + ["t1ce", "mask"], image_only=True),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            # 随机裁块用于训练
            RandCropByPosNegLabeld(keys=keys + ["t1ce", "mask"], label_key="mask", spatial_size=crop_size, pos=3, neg=1,
                                   num_samples=num_samples),
            # # 固定裁块
            # CenterSpatialCropd(keys=keys+["t1ce","mask"],roi_size=(80,80,16)),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
            # 随机旋转
            RandRotated(keys=["image", "t1ce", "mask"],
                        range_x=30 * math.pi / 180,
                        range_y=30 * math.pi / 180,
                        range_z=30 * math.pi / 180,
                        prob=random_prob,
                        padding_mode="zeros"),
            # 随机翻转
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                prob=random_prob,
                spatial_axis=[0]),
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                spatial_axis=[1],
                prob=random_prob,
            ),
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                spatial_axis=[2],
                prob=random_prob,
            ),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-175,
            #     a_max=250,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            #
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),
        ]
    )
    return train_transforms


def get_3d_train_transform(keys, random_prob):
    train_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            DivisiblePadd(keys=keys + ["t1ce", "mask"], k=16, mode="reflect"),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
            # 随机旋转
            RandRotated(keys=["image", "t1ce", "mask"],
                        range_x=30 * math.pi / 180,
                        range_y=30 * math.pi / 180,
                        range_z=30 * math.pi / 180,
                        prob=random_prob,
                        padding_mode="reflection"),
            # 随机翻转
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                prob=random_prob,
                spatial_axis=[0]),
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                spatial_axis=[1],
                prob=random_prob,
            ),
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                spatial_axis=[2],
                prob=random_prob,
            ),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-175,
            #     a_max=250,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            #
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),
        ]
    )
    return train_transforms


def get_2d_train_transform(keys, random_prob):
    train_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask", "prostate_mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask", "prostate_mask"], channel_dim="no_channel"),
            # padding 到可被32整除
            DivisiblePadd(keys=keys + ["t1ce", "mask", "prostate_mask"], k=32, mode="reflect"),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask", "prostate_mask"]),
            # 随机旋转
            RandRotated(keys=["image", "t1ce", "mask", "prostate_mask"],
                        range_x=30 * math.pi / 180,
                        range_y=30 * math.pi / 180,
                        prob=random_prob,
                        padding_mode="reflection",
                        mode=["bilinear", "bilinear", "nearest", "nearest"]
                        ),
            # 随机翻转
            RandFlipd(
                keys=["image", "t1ce", "mask", "prostate_mask"],
                prob=random_prob,
                spatial_axis=[0],
            ),
            RandFlipd(
                keys=["image", "t1ce", "mask", "prostate_mask"],
                spatial_axis=[1],
                prob=random_prob,
            ),
            # RandZoomd(keys=["image", "t1ce"], min_zoom= 0.8, max_zoom=1.3, prob=random_prob, mode="bilinear", padding_mode="constant"),
            # #
            # RandScaleIntensityd(keys=["image", "t1ce"], factors=0.5, prob=random_prob),
            # RandShiftIntensityd(keys=["image", "t1ce"], offsets=0.2,prob=random_prob),
            # RandAdjustContrastd(keys=["image", "t1ce"], gamma=(0.7,1.3), prob=random_prob),
            #
            # RandGaussianNoised(keys=["image"], prob=0.1),
            # RandGaussianNoised(keys=["t1ce"], prob=0.1)
        ]
    )
    return train_transforms


def get_2d_train_transform_diff(keys, random_prob):
    train_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            # padding 到可被32整除
            DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32, mode="reflect"),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
            # 随机旋转
            RandRotated(keys=["image", "t1ce", "mask"],
                        range_x=30 * math.pi / 180,
                        range_y=30 * math.pi / 180,
                        prob=random_prob,
                        padding_mode="reflection",
                        mode=["bilinear", "bilinear", "nearest"]
                        ),
            # 随机翻转
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                prob=random_prob,
                spatial_axis=[0],
            ),
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                spatial_axis=[1],
                prob=random_prob,
            ),
            # RandZoomd(keys=["image", "t1ce"], min_zoom= 0.8, max_zoom=1.3, prob=random_prob, mode="bilinear", padding_mode="constant"),
            # #
            # RandScaleIntensityd(keys=["image", "t1ce"], factors=0.5, prob=random_prob),
            # RandShiftIntensityd(keys=["image", "t1ce"], offsets=0.2,prob=random_prob),
            # RandAdjustContrastd(keys=["image", "t1ce"], gamma=(0.7,1.3), prob=random_prob),
            #
            # RandGaussianNoised(keys=["image"], prob=0.1),
            # RandGaussianNoised(keys=["t1ce"], prob=0.1)
        ]
    )
    return train_transforms


def get_2d_rgb_train_transform(keys, random_prob):
    train_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            RepeatChanneld(keys=keys + ["t1ce", "mask"], repeats=3),
            # padding 到可被32整除
            DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32, mode="reflect"),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
            # 随机旋转
            RandRotated(keys=["image", "t1ce", "mask"],
                        range_x=30 * math.pi / 180,
                        range_y=30 * math.pi / 180,
                        prob=random_prob,
                        padding_mode="reflection"
                        ),
            # 随机翻转
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                prob=random_prob,
                spatial_axis=[0]),
            RandFlipd(
                keys=["image", "t1ce", "mask"],
                spatial_axis=[1],
                prob=random_prob,
            ),
            # RandZoomd(keys=["image", "t1ce"], min_zoom= 0.8, max_zoom=1.3, prob=random_prob, mode="bilinear", padding_mode="constant"),
            # #
            # RandScaleIntensityd(keys=["image", "t1ce"], factors=0.5, prob=random_prob),
            # RandShiftIntensityd(keys=["image", "t1ce"], offsets=0.2,prob=random_prob),
            # RandAdjustContrastd(keys=["image", "t1ce"], gamma=(0.7,1.3), prob=random_prob),
            #
            # RandGaussianNoised(keys=["image"], prob=0.1),
            # RandGaussianNoised(keys=["t1ce"], prob=0.1)
        ]
    )
    return train_transforms


def get_val_transforms(keys, num_samples, crop_size):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadImaged(keys=keys + ["t1ce", "mask"], image_only=True),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            RandCropByPosNegLabeld(keys=keys + ["t1ce", "mask"], label_key="mask", spatial_size=crop_size, pos=3, neg=1,
                                   num_samples=num_samples),
            # # 固定裁块
            # CenterSpatialCropd(keys=keys + ["t1ce", "mask"], roi_size=(80, 80, 16)),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            # totensor
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms


def get_3d_val_transform(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            DivisiblePadd(keys=keys + ["t1ce", "mask"], k=16, mode="reflect"),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            # totensor
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms


def get_2d_val_transform(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask", "prostate_mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask", "prostate_mask"], channel_dim="no_channel"),
            # padding 到可被32整除
            DivisiblePadd(keys=keys + ["t1ce", "mask", "prostate_mask"], k=32),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask", "prostate_mask"]),
        ]
    )
    return val_transforms


def get_2d_val_transform_diff(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            # padding 到可被32整除
            DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms


def get_2d_rgb_val_transform(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            RepeatChanneld(keys=keys + ["t1ce", "mask"], repeats=3),
            # padding 到可被32整除
            DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms


def get_test_transforms(keys):
    test_transforms = Compose(
        [
            # 固定的transform
            LoadImaged(keys=keys + ["t1ce", "mask"], image_only=True),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            # # 固定裁块
            # CenterSpatialCropd(keys=keys + ["t1ce", "mask"], roi_size=(80, 80, 16)),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            # totensor
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return test_transforms


def get_2d_test_transform(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            # padding 到可被32整除
            # DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms


def get_2d_rgb_test_transform(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            RepeatChanneld(keys=keys + ["t1ce", "mask"], repeats=3),
            # padding 到可被32整除
            # DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms


def get_wholebody_test_transform(keys):
    val_transforms = Compose(
        [
            # 固定的transform
            # LoadImaged(keys=keys + ["t1ce", "mask"],reader="ITKReader"),
            LoadImageITKd(keys=keys + ["t1ce", "mask"]),
            EnsureChannelFirstd(keys=keys + ["t1ce", "mask"], channel_dim="no_channel"),
            # padding 到可被32整除
            # DivisiblePadd(keys=keys + ["t1ce", "mask"], k=32),
            # 拼接,keys变成image
            ConcatItemsd(keys=keys, name="image", dim=0),
            ToTensord(keys=["image", "t1ce", "mask"]),
        ]
    )
    return val_transforms
