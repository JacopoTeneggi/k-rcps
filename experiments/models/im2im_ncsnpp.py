from models.ncsnpp import NCSNpp, conv3x3
from models.im2im.add_uncertainty import ModelWithUncertainty
from models.im2im.finallayers.quantile_layer import (
    QuantileRegressionLayer,
    quantile_regression_loss_fn,
    quantile_regression_nested_sets_from_output,
)
from models.utils import register_model


@register_model(name="im2im_ncsnpp")
class im2imNCSNpp(ModelWithUncertainty):
    def __init__(self, config):
        self.config = config
        # Initialize base model (NCSNpp)
        model = NCSNpp(config)
        # replace last layer to 3x3 conv to 32 channels
        n_channels_in = model.all_modules[-1].in_channels
        n_channels_middle = 32
        n_channels_out = config.data.num_channels
        model.all_modules[-1] = conv3x3(
            n_channels_in,
            n_channels_middle,
            bias=True,
            init_scale=config.model.init_scale,
        )
        # initialize quantile regression layer
        last_layer = QuantileRegressionLayer(
            n_channels_middle,
            n_channels_out,
            config.qr_params,
        )
        # initialize model with uncertainty
        train_loss_fn = quantile_regression_loss_fn
        nested_sets_from_output_fn = quantile_regression_nested_sets_from_output
        super(im2imNCSNpp, self).__init__(
            model,
            last_layer,
            train_loss_fn,
            nested_sets_from_output_fn,
            config.qr_params,
        )

    def forward(self, x, time_cond):
        x = self.baseModel(x, time_cond)
        x = self.last_layer(x)
        return x
