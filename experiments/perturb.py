import os
import sys
import ml_collections
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader
from dataset import get_dataset
from tqdm import tqdm

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Data Configuration", lock_config=False)
flags.DEFINE_string("workdir", "./", "Working directory")


def main(_):
    workdir = FLAGS.workdir
    denoising_dir = os.path.join(workdir, "denoising")

    data_config = FLAGS.config
    if data_config.dataset == "CelebA":
        data_config.image_size, sigma0 = 128, 1.0
    if data_config.dataset == "AbdomenCT-1K":
        data_config.image_size, sigma0 = 512, 0.4
    data_config.return_img_id = True

    config = ml_collections.ConfigDict()
    config.data = data_config

    _, dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

    denoising_dataset_dir = os.path.join(denoising_dir, config.data.name)
    perturbed_dir = os.path.join(denoising_dataset_dir, "perturbed")
    os.makedirs(perturbed_dir, exist_ok=True)

    sigma0 = torch.tensor([sigma0])
    for _, data in enumerate(tqdm(dataloader)):
        input, img_id = data

        z = torch.randn_like(input)
        perturbed = input + sigma0[:, None, None, None] * z

        for _id, _perturbed in zip(img_id, perturbed):
            torch.save(_perturbed, os.path.join(perturbed_dir, f"{_id}.pt"))


if __name__ == "__main__":
    app.run(main)
