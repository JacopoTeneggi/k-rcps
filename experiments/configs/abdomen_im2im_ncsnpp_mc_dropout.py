import os
from configs import abdomen_im2im_ncsnpp, utils


@utils.register_config(name="abdomen_im2im_ncsnpp_mc_dropout")
def get_config():
    config = utils.get_config(name="default_abdomen")
    config.name = name = os.path.basename(__file__.split(".")[0])
    config.deepspeed_config = os.path.join("configs", "ds", f"{name}.json")

    # model
    model = config.model
    model.name = "mc_dropout"
    model.im2im_config = utils.get_config(name="abdomen_im2im_ncsnpp")

    return config
