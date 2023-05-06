import torch
import torch.nn as nn

from models import ncsnpp
from models.utils import register_model


@register_model(name="quantile_ncsnpp")
class QuantileNCSNpp(nn.Module):
    def __init__(self, config):
        super(QuantileNCSNpp, self).__init__()
        self.config = config

        ncsnpp_model_config = config.model
        init_scale = ncsnpp_model_config.init_scale
        ncsnpp_model_config.scale_by_sigma = False
        self.ncsnpp = ncsnpp.NCSNpp(config)

        in_ch = self.ncsnpp.all_modules[-1].in_channels
        self.ncsnpp.all_modules[-1] = ncsnpp.conv3x3(
            in_ch, in_ch, bias=True, init_scale=init_scale
        )

        out_channels = config.data.num_channels
        self.lower = ncsnpp.conv3x3(
            in_ch, out_channels, bias=True, init_scale=init_scale
        )
        self.prediction = ncsnpp.conv3x3(
            in_ch, out_channels, bias=True, init_scale=init_scale
        )
        self.upper = ncsnpp.conv3x3(
            in_ch, out_channels, bias=True, init_scale=init_scale
        )

    def forward(self, x, time_cond):
        h = self.ncsnpp(x, time_cond)
        h = torch.cat(
            [
                self.lower(h).unsqueeze(1),
                self.prediction(h).unsqueeze(1),
                self.upper(h).unsqueeze(1),
            ],
            dim=1,
        )
        time_cond = time_cond.reshape((h.shape[0], *([1] * len(h.shape[1:]))))
        h = h / time_cond
        return h
