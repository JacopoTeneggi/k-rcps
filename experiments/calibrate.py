import os
import torch
import numpy as np
from absl import app, flags
from ml_collections.config_flags import config_flags
from configs import (
    celeba_ncsnpp,
    abdomen_ncsnpp,
    celeba_ncsnpp_conffusion,
    abdomen_ncsnpp_conffusion,
    celeba_im2im_ncsnpp,
    abdomen_im2im_ncsnpp,
    celeba_im2im_ncsnpp_mc_dropout,
    abdomen_im2im_ncsnpp_mc_dropout,
)
from krcps.utils import (
    get_uq,
    get_loss,
    get_calibration,
    _split_idx,
)
from utils import denoising_results
from dataset import get_dataset
from tqdm import tqdm

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration", lock_config=True)
flags.DEFINE_string("id", None, "Experiment Unique ID")
flags.DEFINE_string("workdir", "./", "Working directory")


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir

    r = 20
    n_cal, n_val = None, 128
    if config.data.dataset == "CelebA":
        alpha = 0.10
        delta = epsilon = 0.10

        dlambda = 5e-03
    if config.data.dataset == "AbdomenCT1K":
        alpha = 0.20
        delta = 0.1
        epsilon = 0.05

        dlambda = 2e-03

    calibration_dir = os.path.join(workdir, "calibration")
    calibration_dataset_dir = os.path.join(calibration_dir, config.data.dataset)
    calibration_dir = os.path.join(calibration_dataset_dir, config.name)
    os.makedirs(calibration_dir, exist_ok=True)

    N = 300
    _, dataset = get_dataset(config)
    (_, original, _, denoised) = denoising_results(dataset, config, n=n_cal)

    if config.data.dataset == "CelebA":
        original = torch.mean(original, dim=1)
        denoised = torch.mean(denoised, dim=1)
    if config.data.dataset == "AbdomenCT1K":
        original = original.squeeze()
        denoised = denoised.squeeze()

    rcps_loss_fn = get_loss("vector_01")

    def _calibrate(calibration_name, uq_name, uq_fn, calibration_fn):
        print(
            f"Calibrating {uq_name} with {calibration_name} (epsilon = {epsilon}, delta = {delta})"
        )

        n = original.size(0)

        r_val_idx = []
        r_lambda = []
        r_val_loss = []
        r_val_l = []
        r_val_u = []
        r_val_i = []
        for _ in tqdm(range(r)):
            val_idx, cal_idx = _split_idx(n, n_val)

            cal_original, cal_denoised = original[cal_idx], denoised[cal_idx]
            val_original, val_denoised = original[val_idx], denoised[val_idx]

            cal_I = uq_fn(cal_denoised)
            val_I = uq_fn(val_denoised)

            _lambda = calibration_fn(cal_original, cal_I)

            val_l, val_u = val_I(_lambda)
            val_loss = rcps_loss_fn(val_original, val_l, val_u)
            val_i = val_u - val_l
            print(torch.mean(val_i))

            r_val_idx.append(torch.tensor(val_idx))
            r_lambda.append(_lambda)
            r_val_loss.append(val_loss)
            r_val_l.append(val_l)
            r_val_u.append(val_u)
            r_val_i.append(val_i)

        r_val_idx = torch.stack(r_val_idx)
        r_lambda = torch.stack(r_lambda)
        r_val_loss = torch.stack(r_val_loss)
        r_val_l = torch.stack(r_val_l)
        r_val_u = torch.stack(r_val_u)
        r_val_i = torch.stack(r_val_i)

        _calibration_dir = os.path.join(calibration_dir, calibration_name)
        os.makedirs(_calibration_dir, exist_ok=True)
        torch.save(r_val_idx, os.path.join(_calibration_dir, f"{uq_name}_val_idx.pt"))
        torch.save(r_lambda, os.path.join(_calibration_dir, f"{uq_name}_lambda.pt"))
        torch.save(r_val_loss, os.path.join(_calibration_dir, f"{uq_name}_val_loss.pt"))
        torch.save(r_val_l, os.path.join(_calibration_dir, f"{uq_name}_val_l.pt"))
        torch.save(r_val_u, os.path.join(_calibration_dir, f"{uq_name}_val_u.pt"))
        torch.save(r_val_i, os.path.join(_calibration_dir, f"{uq_name}_val_i.pt"))

    def _calibrate_rcps(uq_name, uq_fn, lambda_max):
        rcps_fn = get_calibration("rcps")

        def _rcps_calibration(cal_original, cal_I):
            return rcps_fn(
                cal_original,
                cal_I,
                "01",
                "hoeffding_bentkus",
                epsilon,
                delta,
                lambda_max,
                dlambda,
            )

        _calibrate("rcps", uq_name, uq_fn, _rcps_calibration)

    def _calibrate_k_rcps(uq_name, uq_fn, lambda_max):
        k_rcps_fn = get_calibration("k_rcps")

        for n_opt in [128, 256]:
            for prob_size in [50, 100]:
                for k in [4, 8, 32]:
                    calibration_name = f"k_rcps_{n_opt}_{prob_size}_{k}"

                    def _k_rcps_calibration(cal_original, cal_I):
                        return k_rcps_fn(
                            cal_original,
                            cal_I,
                            "hoeffding_bentkus",
                            epsilon,
                            delta,
                            lambda_max,
                            dlambda,
                            k,
                            "01_loss_quantile",
                            n_opt,
                            prob_size,
                            gamma=np.linspace(0.3, 0.7, 16),
                        )

                    _calibrate(calibration_name, uq_name, uq_fn, _k_rcps_calibration)

    if config.model.name == "ncsnpp":
        uq_name, lambda_max = "naive_sampling_additive", 0.2
        uq_fn = get_uq(uq_name, alpha=alpha, dim=1)

        _calibrate_rcps(uq_name, uq_fn, lambda_max)
        _calibrate_k_rcps(uq_name, uq_fn, lambda_max)

        uq_name, lambda_max = "calibrated_quantile", 0.6
        uq_fn = get_uq(uq_name, alpha=alpha, dim=1)

        _calibrate_rcps(uq_name, uq_fn, lambda_max)
        _calibrate_k_rcps(uq_name, uq_fn, lambda_max)
    if config.model.name == "im2im_ncsnpp":
        uq_name, lambda_max = "quantile_regression", 1.2
        uq_fn = get_uq(uq_name)

        _calibrate_rcps(uq_name, uq_fn, lambda_max)
    if config.model.name == "conffusion":
        uq_name, lambda_max = (
            "conffusion_multiplicative",
            1.2 if config.data.dataset == "AbdomenCT1K" else 4.0,
        )
        uq_fn = get_uq(uq_name)

        _calibrate_rcps(uq_name, uq_fn, lambda_max)

        uq_name, lambda_max = (
            "conffusion_additive",
            0.2 if config.data.dataset == "AbdomenCT1K" else 0.6,
        )
        uq_fn = get_uq(uq_name)

        _calibrate_rcps(uq_name, uq_fn, lambda_max)
        _calibrate_k_rcps(uq_name, uq_fn, lambda_max)
    if config.model.name == "mc_dropout":
        uq_name, lambda_max = (
            "std",
            21.0 if config.data.dataset == "AbdomenCT1K" else 6.0,
        )
        uq_fn = get_uq(uq_name, dim=1)

        _calibrate_rcps(uq_name, uq_fn, lambda_max)


if __name__ == "__main__":
    app.run(main)
