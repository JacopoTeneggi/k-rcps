import os
import ml_collections
from configs import abdomen_ncsnpp, utils


@utils.register_config(name="abdomen_im2im_ncsnpp")
def get_config():
    config = utils.get_config(name="abdomen_ncsnpp")
    config.name = name = os.path.basename(__file__.split(".")[0])
    config.deepspeed_config = os.path.join("configs", "ds", f"{name}.json")

    # model
    model = config.model
    model.name = "im2im_ncsnpp"

    # quantile regression
    config.qr_params = qr_params = ml_collections.ConfigDict()
    qr_params.q_lo = 0.05
    qr_params.q_hi = 0.95
    qr_params.q_lo_weight = 1
    qr_params.q_hi_weight = 1
    qr_params.mse_weight = 1

    return config
