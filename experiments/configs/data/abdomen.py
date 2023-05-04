import ml_collections
from configs.utils import register_data_config


@register_data_config(name="abdomen")
def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = "AbdomenCT-1K"
    config.name = "abdomen"
    config.num_channels = 1
    config.augmentation = True
    config.return_img_id = False
    config.centered = False

    return config
