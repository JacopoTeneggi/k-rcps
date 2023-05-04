import ml_collections
from configs.utils import register_data_config


@register_data_config(name="celeba")
def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = "CelebA"
    config.name = "celeba"
    config.num_channels = 3
    config.augmentation = True
    config.return_img_id = False
    config.centered = False

    return config
