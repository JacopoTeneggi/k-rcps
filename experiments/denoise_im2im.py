import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import app, flags
from ml_collections.config_flags import config_flags
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dataset import get_dataset
from models import im2im_ncsnpp
from models import utils as mutils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration", lock_config=True)
flags.DEFINE_string("workdir", "./", "Working directory")
flags.DEFINE_string("gpu", "0,1,2,3,4,5,6,7", "GPU(s) to use")


def common(device, world_size, config, workdir, batch_size=1):
    # setup process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=device, world_size=world_size)
    torch.cuda.set_device(device)

    # setup dirs
    denoising_dir = os.path.join(workdir, "denoising")
    denoising_dataset_dir = os.path.join(denoising_dir, config.data.name)
    perturbed_dir = os.path.join(denoising_dataset_dir, "perturbed")
    denoised_dir = os.path.join(os.path.join(denoising_dataset_dir, config.name))
    os.makedirs(denoised_dir, exist_ok=True)

    # setup dataset
    config.data.return_img_id = True
    _, dataset = get_dataset(config)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    if config.data.dataset == "CelebA":
        sigma0 = 1.0
    if config.data.dataset == "AbdomenCT-1K":
        sigma0 = 0.4

    # setup model
    model = mutils.get_model(config, checkpoint=True)
    model = model.to(device)
    model.eval()
    model = DDP(model, device_ids=[device], output_device=device)
    torch.set_grad_enabled(False)

    return dataloader, sigma0, model, perturbed_dir, denoised_dir


def denoise(rank, world_size, config, workdir):
    device = rank
    dataloader, sigma0, model, perturbed_dir, denoised_dir = common(
        device, world_size, config, workdir, batch_size=4
    )

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


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir
    gpu = FLAGS.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    world_size = len(gpu.split(","))

    mp.spawn(denoise, args=(world_size, config, workdir), nprocs=world_size)


if __name__ == "__main__":
    app.run(main)
