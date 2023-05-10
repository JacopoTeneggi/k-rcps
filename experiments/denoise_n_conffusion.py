import os
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader
from dataset import get_dataset
from models import utils as mutils
from tqdm import tqdm

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration", lock_config=False)
flags.DEFINE_string("gpu", "0", "GPU to use")
flags.DEFINE_string("workdir", "./", "Working directory")


def main(_):
    config = FLAGS.config
    gpu = FLAGS.gpu
    workdir = FLAGS.workdir

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    denoising_dir = os.path.join(workdir, "denoising")
    denoising_dataset_dir = os.path.join(denoising_dir, config.data.name)
    perturbed_dir = os.path.join(denoising_dataset_dir, "perturbed")
    denoised_dir = os.path.join(denoising_dataset_dir, config.name)
    os.makedirs(denoised_dir, exist_ok=True)

    batch_size = 4
    config.data.return_img_id = True
    _, dataset = get_dataset(config)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    if config.data.dataset == "CelebA":
        sigma0 = 1.0
    if config.data.dataset == "AbdomenCT1K":
        sigma0 = 0.4

    model = mutils.get_model(config, checkpoint=True)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    for _, data in enumerate(tqdm(dataloader)):
        _, img_id = data

        y = torch.stack(
            [torch.load(os.path.join(perturbed_dir, f"{_id}.pt")) for _id in img_id],
            dim=0,
        )
        y = y.to(device)

        sigma_t = torch.tensor(y.size(0) * [sigma0], device=device)

        output = model(y, sigma_t)

        output = output.cpu()
        for _id, _output in zip(img_id, output):
            torch.save(_output[0], os.path.join(denoised_dir, f"{_id}_l.pt"))
            torch.save(_output[1], os.path.join(denoised_dir, f"{_id}.pt"))
            torch.save(_output[2], os.path.join(denoised_dir, f"{_id}_u.pt"))


if __name__ == "__main__":
    app.run(main)
