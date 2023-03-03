from configs.afhq import default_afhq_configs
import numpy as np

# This is the smaller AFHQ model


def get_config():
    config = default_afhq_configs.get_default_configs()
    model = config.model
    config.data.image_size = 64
    # Parameters mostly taken from the Diffusion models beat GANs paper
    model.model_channels = 128
    model.num_heads = 1
    model.num_res_blocks = 2
    model.channel_mult = (1, 2, 3, 4)
    model.attention_levels = (2, 3)
    model.dropout = 0.1
    config.training.batch_size = 128
    config.eval.batch_size = 9
    model.blur_sigma_max = 32
    model.blur_sigma_min = 0.5
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                             np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array(
        [0] + list(model.blur_schedule))  # Add the k=0 timestep
    model.blur_schedule[-1] = 0
    model.blur_rate = 'custom'
    config.eval.num_samples = 10000
    config.training.sampling_freq = 25000
    config.optim.lr = 1e-4
    model.ema_rate = 0.999
    return config
