import os
import torch
import wandb
import torchvision.transforms as t
from absl import app, flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader
from dataset import CelebA, AbdomenCT1K
from models.im2im.finallayers.quantile_layer import (
    quantile_regression_loss_fn,
)
from models import ncsnpp_conffusion
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

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_dir = os.path.join(workdir, "data", config.data.dataset)

    augmentation_t = t.Compose(
        [
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
            t.RandomRotation(90),
        ]
    )
    if config.data.dataset == "CelebA":
        sigma0 = 1.0

        _t = t.Compose(
            [
                t.ToTensor(),
                t.CenterCrop((config.data.image_size, config.data.image_size)),
                t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        _t = t.Compose([_t, augmentation_t])

        dataset = CelebA(
            data_dir,
            split="test",
            target_type=[],
            transform=_t,
            download=False,
        )
    if config.data.dataset == "AbdomenCT1K":
        sigma0 = 0.4

        _t = t.Compose(
            [t.ToTensor(), t.Resize((config.data.image_size, config.data.image_size))]
        )
        _t = t.Compose([_t, augmentation_t])

        dataset = AbdomenCT1K(
            data_dir,
            op="finetuning",
            transform=_t,
        )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model = mutils.get_model(config)
    model = model.to(device)
    model.train()

    criterion = quantile_regression_loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

    wandb.init(project="n_conffusion", entity="jacopoteneggi")

    n_epochs = 10

    running_loss = 0.0

    sigma0 = torch.tensor([sigma0], device=device)
    for _ in range(n_epochs):
        for i, x in enumerate(tqdm(dataloader)):
            x = x.to(device)

            z = torch.randn_like(x)
            y = x + sigma0[:, None, None, None] * z

            output = model(y, sigma0.repeat(y.size(0)))

            loss = criterion(output, x, model.qr_params)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_step = 5
            if (i + 1) % log_step == 0:
                wandb.log({"train_loss": running_loss / log_step})
                running_loss = 0.0

        torch.save(
            model.state_dict(), os.path.join(checkpoint_dir, f"{config.name}.pt")
        )


if __name__ == "__main__":
    app.run(main)
