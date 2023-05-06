import os
import deepspeed as ds
import numpy as np
import torch
import wandb
from absl import app, flags
from ml_collections.config_flags import config_flags
from dataset import get_dataset
from models import ncsnpp
from models import utils as mutils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration", lock_config=True)
flags.DEFINE_string("workdir", "./", "Working directory")
flags.DEFINE_string("include", None, "Device to include")
flags.DEFINE_integer("local_rank", -1, "Local rank passed from distributed launcher")
flags.DEFINE_integer("master_port", 29500, "Master port for distributed training")


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize dataset
    train_dataset, _ = get_dataset(config)

    # Initialize model
    model = mutils.get_model(config)
    model_engine, _, train_loader, _ = ds.initialize(
        args=config,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
    )
    model_engine.train()

    # Initialize SDE
    eps = config.model.eps
    sigma_min = config.model.sigma_min
    sigma_max = config.model.sigma_max

    sigma = lambda t: sigma_min * (sigma_max / sigma_min) ** t
    diffusion = lambda t: sigma(t) * torch.sqrt(
        torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device)
    )

    # Initialize loss functions
    def target_fn(input):
        # sample time
        T = 1
        t = torch.rand(input.size(0), device=input.device) * (T - eps) + eps
        sigma_t = sigma(t)

        # sample noise
        z = torch.randn_like(input)

        # perturb input
        x = input + sigma_t[:, None, None, None] * z
        target = -z / sigma_t[:, None, None, None]
        return x, t, target

    def score_loss_fn(pred, target):
        return torch.square(pred - target)

    score_fn = lambda x, t: model_engine(x, sigma(t))

    def loss_fn(input):
        # perturb input
        x, t, target = target_fn(input)

        # compute score
        score = score_fn(x, t)

        # compute score loss
        score_loss = score_loss_fn(score, target)
        score_loss = score_loss.reshape((input.size(0), -1))
        loss = 1 / 2 * torch.sum(score_loss, dim=-1) * (diffusion(t) ** 2)

        loss = torch.mean(loss)
        return loss

    wandb.init(project="uncertainty", entity="jacopoteneggi")

    step = 0
    running_loss = 0.0
    while step < config.training.n_iters:
        for data in train_loader:
            input = data

            input = input.to(model_engine.device)
            loss = loss_fn(input)

            running_loss += loss

            model_engine.backward(loss)
            model_engine.step()
            step += 1

            if (step % config.training.log_freq) == 0:
                wandb.log(
                    {
                        "step": step,
                        "train_loss": running_loss / config.training.log_freq,
                    }
                )
                running_loss = 0.0

            if (step % config.training.checkpoint_freq) == 0:
                model_engine.save_checkpoint(
                    checkpoint_dir, tag=f"{config.name}_step_{step}"
                )

    model_engine.save_checkpoint(checkpoint_dir, tag=f"{config.name}_final")


if __name__ == "__main__":
    app.run(main)
