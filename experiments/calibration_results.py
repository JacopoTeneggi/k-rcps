import os
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Data Configuration", lock_config=False)
flags.DEFINE_string("workdir", "./", "Working directory")


def main(_):
    data_config = FLAGS.config

    calibration_dict = {
        "ncsnpp": [
            {"name": "rcps", "uq_name": ["naive_sampling_additive"]},
            {"name": "krcps", "uq_name": ["naive_sampling_additive"]},
        ],
        "ncsnpp_conffusion": [
            {
                "name": "rcps",
                "uq_name": [
                    "conffusion_multiplicative",
                    "conffusion_additive",
                ],
            },
            {"name": "krcps", "uq_name": ["conffusion_additive"]},
        ],
        "im2im_ncsnpp": [
            {"name": "rcps", "uq_name": ["quantile_regression"]},
        ],
        "im2im_ncsnpp_mc_dropout": [{"name": "rcps", "uq_name": ["std"]}],
    }

    dataset = data_config.name
    dataset_calibration_dir = os.path.join("calibration", dataset)

    def _val_i_mu_std(val_i):
        assert val_i.size(0) == 20
        val_i = val_i.view(val_i.size(0), -1)
        r_mu = torch.mean(val_i, dim=1)
        mu, std = torch.mean(r_mu).item(), torch.std(r_mu).item()
        return mu, std

    for model_name, calibration in calibration_dict.items():
        config_name = f"{dataset}_{model_name}"
        config_calibration_dir = os.path.join(dataset_calibration_dir, config_name)

        for _calibration in calibration:
            calibration_name = _calibration["name"]
            uq_name = _calibration["uq_name"]

            for _uq_name in uq_name:
                if calibration_name == "rcps":
                    calibration_dir = os.path.join(
                        config_calibration_dir, calibration_name
                    )

                    val_i = torch.load(
                        os.path.join(calibration_dir, f"{_uq_name}_val_i.pt")
                    )
                    mu, std = _val_i_mu_std(val_i)

                if calibration_name == "krcps":
                    results = []
                    for n_opt in [128, 256]:
                        for d in [50, 100]:
                            for k in [4, 8, 32]:
                                k_rcps_name = f"k_rcps_{n_opt}_{d}_{k}"
                                calibration_dir = os.path.join(
                                    config_calibration_dir, k_rcps_name
                                )

                                try:
                                    val_i = torch.load(
                                        os.path.join(
                                            calibration_dir, f"{_uq_name}_val_i.pt"
                                        )
                                    )
                                    results.append(_val_i_mu_std(val_i))
                                except:
                                    print(f"Skipping: {k_rcps_name}")

                    results = sorted(results, key=lambda x: x[0])
                    mu, std = results[0]

                print(
                    f"config_name: {config_name}, uq_name: {_uq_name}, calibration_name: {calibration_name}, mu: {mu:.4f}, std: {std:.4f}"
                )


if __name__ == "__main__":
    app.run(main)
