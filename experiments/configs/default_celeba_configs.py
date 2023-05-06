import ml_collections
from configs import utils
from configs.data import celeba


@utils.register_config(name="default_celeba")
def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.n_iters = 400000
    training.checkpoint_freq = 50000
    training.log_freq = 50
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # data
    config.data = data = utils.get_data_config(name="celeba")
    data.image_size = 128

    # model
    config.model = model = ml_collections.ConfigDict()
    model.eps = 1e-05
    model.sigma_min = 0.01
    model.sigma_max = 90
    model.num_scales = 300
    model.T = 1.0
    model.sampling_batch_size = 32
    model.total_samples = 128
    model.dropout = 0.1
    model.embedding_type = "fourier"

    return config
