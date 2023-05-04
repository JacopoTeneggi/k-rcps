import os
from . import celeba_ncsnpp, utils


@utils.register_config(name="celeba_ncsnpp_conffusion")
def get_config():
    config = utils.get_config(name="celeba_ncsnpp")
    config.name = name = os.path.basename(__file__.split(".")[0])
    config.deepspeed_config = os.path.join("configs", "ds", f"{name}.json")

    # model
    model = config.model
    model.name = "conffusion"

    return config
