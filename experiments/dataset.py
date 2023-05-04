import os
import numpy as np
import torchvision.datasets as d
import torchvision.transforms as t
from torch.utils.data import Dataset

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class _struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_dataset(config=None, **kwargs):
    if config is None:
        config = _struct(**kwargs)

    data_dir = os.path.join(_DATA_DIR, config.data.dataset)

    augmentation_t = t.Compose(
        [
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
            t.RandomRotation(90),
        ]
    )

    if config.data.dataset == "CelebA":
        train_t = resize_t = t.Compose(
            [
                t.ToTensor(),
                t.CenterCrop((config.data.image_size, config.data.image_size)),
                t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if config.data.augmentation:
            train_t = t.Compose([train_t, augmentation_t])

        train_op, calibration_op, target_type = "train", "valid", []

        train_dataset = CelebA(
            root=data_dir,
            split=train_op,
            target_type=target_type,
            transform=train_t,
            download=False,
            return_img_id=config.data.return_img_id,
        )
        calibration_dataset = CelebA(
            root=data_dir,
            split=calibration_op,
            target_type=target_type,
            transform=resize_t,
            download=False,
            return_img_id=config.data.return_img_id,
        )

    if config.data.dataset == "AbdomenCT-1K":
        train_t = resize_t = t.Compose(
            [t.ToTensor(), t.Resize((config.data.image_size, config.data.image_size))]
        )
        if config.data.augmentation:
            train_t = t.Compose([train_t, augmentation_t])

        train_op, calibration_op = "train", "calibration"

        train_dataset = AbdomenCT1K(
            data_dir,
            op=train_op,
            transform=train_t,
            return_img_id=config.data.return_img_id,
        )
        calibration_dataset = AbdomenCT1K(
            data_dir,
            op=calibration_op,
            transform=resize_t,
            return_img_id=config.data.return_img_id,
        )
    return train_dataset, calibration_dataset


class CelebA(d.CelebA):
    def __init__(self, *args, return_img_id=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_img_id = return_img_id
        if kwargs["split"] == "valid":
            self.op = "calibration"
            l = 768
            self.filename = self.filename[:l]
            self.identity = self.identity[:l]
            self.bbox = self.bbox[:l]
            self.landmarks_align = self.landmarks_align[:l]
            self.attr = self.attr[:l]
        self.ids = self._ids()

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        if self.return_img_id:
            return img, idx
        else:
            return img

    def _ids(self):
        return list(range(len(self)))


class AbdomenCT1K(Dataset):
    ww = 400
    wl = 40

    def __init__(self, root, op, transform, return_img_id=False):
        self.root = root
        self.op = op
        if op == "train":
            self.op_images = np.load(
                os.path.join(root, "train_images.npy"), allow_pickle=True
            )
        if op == "calibration":
            l_cal = 512
            l_test = 128
            cal_images = np.load(
                os.path.join(root, "calibration_images.npy"), allow_pickle=True
            )
            test_images = np.load(
                os.path.join(root, "test_images.npy"), allow_pickle=True
            )

            cal_images = cal_images[:l_cal]
            test_images = test_images[:l_test]
            self.op_images = np.concatenate((cal_images, test_images))
        self.transform = transform
        self.return_img_id = return_img_id
        self.ids = self._ids()

    def rescale_transform(self, x):
        image_min = self.wl - self.ww // 2
        image_max = self.wl + self.ww // 2

        mu = image_min
        std = image_max - image_min

        return (x * std) + mu

    def __len__(self):
        return len(self.op_images)

    def __getitem__(self, idx):
        img_id = self.op_images[idx]

        img = np.load(
            os.path.join(self.root, "images", f"{img_id}.npy"), allow_pickle=True
        )

        img = self.transform(img)
        img = img.float()

        if self.return_img_id:
            return img, img_id
        else:
            return img

    def _ids(self):
        return list(filter(lambda u: u.split(".")[0], self.op_images.tolist()))
