import ml_collections
import torch
import numpy as np


def get_config():
    return get_default_configs()


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.n_iters = 1300001
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 10000
    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    eval.batch_size = 9
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.calculate_fids = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'lsun_church'
    data.image_size = 128
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.K = 400
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.model_channels = 128
    model.channel_mult = (1, 2, 3, 4, 5)
    model.conv_resample = True
    model.num_heads = 1
    model.conditional = True
    model.attention_levels = (2, 3, 4)
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.num_res_blocks = 4
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.skip_rescale = True
    model.blur_sigma_max = 64
    model.blur_sigma_min = 0.5
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                             np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array(
        [0] + list(model.blur_schedule))  # Add the k=0 timestep

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = 'Adam'
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    optim.automatic_mp = True

    config.seed = 42
    config.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
