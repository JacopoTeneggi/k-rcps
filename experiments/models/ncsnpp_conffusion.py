import ml_collections
import torch.nn as nn
from .im2im.finallayers.quantile_layer import QuantileRegressionLayer
from . import utils


@utils.register_model(name="ncsnpp_conffusion")
class Conffusion(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super(Conffusion, self).__init__()
        from .ncsnpp import NCSNpp, conv3x3

        ncsnpp_config = config.model.ncsnpp_config
        score_network = utils.get_model(ncsnpp_config, checkpoint=True)

        n_channels_in = score_network.all_modules[-1].in_channels
        n_channels_middle = 32
        n_channels_out = ncsnpp_config.data.num_channels
        score_network.all_modules[-1] = conv3x3(
            n_channels_in,
            n_channels_middle,
            bias=True,
            init_scale=ncsnpp_config.model.init_scale,
        )

        self.qr_params = qr_params = ml_collections.ConfigDict()
        qr_params.q_lo = 0.05
        qr_params.q_hi = 0.95
        qr_params.q_lo_weight = 1
        qr_params.q_hi_weight = 1
        qr_params.mse_weight = 0

        qr_head = QuantileRegressionLayer(n_channels_middle, n_channels_out, qr_params)

        self.score_network = score_network
        self.n_conffusion_head = qr_head

    def forward(self, x, time_cond):
        x = self.score_network(x, time_cond)
        x = self.n_conffusion_head(x)
        return x
