from configs.ffhq import default_ffhq_configs
import numpy as np

# Config for the model where image resolution = 128x128, and
# maximum blurring effective length scale is 128 as well
# -> the average colour and other characteristics get disentangled

def get_config():
    config = default_ffhq_configs.get_default_configs()
    model = config.model
    config.data.image_size = 128
    # Parameters mostly taken from the Diffusion models beat GANs paper
    model.model_channels = 128
    model.num_heads = 1
    model.num_res_blocks = 2
    model.channel_mult = (1, 2, 3, 4, 5)
    model.attention_levels = (2, 3, 4)
    model.dropout = 0.1
    config.training.batch_size = 32
    config.eval.batch_size = 9
    model.blur_sigma_max = 128
    model.blur_sigma_min = 0.5
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                             np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array(
        [0] + list(model.blur_schedule))  # Add the k=0 timestep
    model.blur_rate = 'custom'
    config.optim.lr = 2e-4
    config.eval.num_samples = 10000
    return config
