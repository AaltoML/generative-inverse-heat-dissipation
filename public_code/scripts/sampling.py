import torch
import numpy as np
import logging
from scripts import datasets


def get_sampling_fn_inverse_heat(config, initial_sample,
                                 intermediate_sample_indices, delta, device,
                                 share_noise=False):
    """ Returns our inverse heat process sampling function. 
    Arguments: 
    initial_sample: Pytorch Tensor with the initial draw from the prior p(u_K)
    intermediate_sample_indices: list of indices to save (e.g., [0,1,2,3...] or [0,2,4,...])
    delta: Standard deviation of the sampling noise
    share_noise: Whether to use the same noises for all elements in the batch
    """
    K = config.model.K

    def sampler(model):

        if share_noise:
            noises = [torch.randn_like(initial_sample[0], dtype=torch.float)[None]
                      for i in range(K)]
        intermediate_samples_out = []

        with torch.no_grad():
            u = initial_sample.to(config.device).float()
            if intermediate_sample_indices != None and K in intermediate_sample_indices:
                intermediate_samples_out.append((u, u))
            for i in range(K, 0, -1):
                vec_fwd_steps = torch.ones(
                    initial_sample.shape[0], device=device, dtype=torch.long) * i
                # Predict less blurry mean
                u_mean = model(u, vec_fwd_steps) + u
                # Sampling step
                if share_noise:
                    noise = noises[i-1]
                else:
                    noise = torch.randn_like(u)
                u = u_mean + noise*delta
                # Save trajectory
                if intermediate_sample_indices != None and i-1 in intermediate_sample_indices:
                    intermediate_samples_out.append((u, u_mean))

            return u_mean, config.model.K, [u for (u, u_mean) in intermediate_samples_out]
    return sampler


def get_sampling_fn_inverse_heat_interpolate(config, initial_sample,
                                             delta, device, num_points):
    """Returns an interpolation between two images, where the interpolation is
    done with the latent noise and the initial sample. Arguments: 
    initial_sample: Two initial states to interpolate over. 
                                    Shape: (2, num_channels, height, width)
    delta: Sampling noise standard deviation
    num_points: Number of point to use in the interpolation 
    """
    assert initial_sample.shape[0] == 2
    shape = initial_sample.shape

    # Linear interpolation between the two input states
    init_input = torch.linspace(1, 0, num_points, device=device)[:, None, None, None] * initial_sample[0][None] + \
        (1-torch.linspace(1, 0, num_points, device=device)
         )[:, None, None, None] * initial_sample[1][None]
    init_input = init_input.to(config.device).float()
    logging.info("init input shape: {}".format(init_input.shape))

    # Get all the noise steps
    noise1 = [torch.randn_like(init_input[0]).to(device)[None]
              for i in range(0, config.model.K)]
    noise1 = torch.cat(noise1, 0)
    noise2 = [torch.randn_like(init_input[0]).to(device)[None]
              for i in range(0, config.model.K)]
    noise2 = torch.cat(noise2, 0)

    # Spherical interpolation between the noise endpoints.
    noise_weightings = torch.linspace(
        0, np.pi/2, num_points, device=device)[None, :, None, None, None]
    noise1 = noise1[:, None, :, :, :]
    noise2 = noise2[:, None, :, :, :]
    noises = torch.cos(noise_weightings) * noise1 + \
        torch.sin(noise_weightings) * noise2

    K = config.model.K

    def sampler(model):
        with torch.no_grad():
            x = init_input.to(config.device).float()
            for i in range(K, 0, -1):
                vec_fwd_steps = torch.ones(
                    num_points, device=device, dtype=torch.long) * i
                x_mean = model(x, vec_fwd_steps) + x
                noise = noises[i-1]
                x = x_mean + noise*delta
            x_sweep = x_mean
            return x_sweep

    return sampler, init_input


def get_initial_sample(config, forward_heat_module, delta, batch_size=None):
    """Take a draw from the prior p(u_K)"""
    trainloader, _ = datasets.get_dataset(config,
                                          uniform_dequantization=config.data.uniform_dequantization,
                                          train_batch_size=batch_size)

    initial_sample = next(iter(trainloader))[0].to(config.device)
    original_images = initial_sample.clone()
    initial_sample = forward_heat_module(initial_sample,
                                         config.model.K * torch.ones(initial_sample.shape[0], dtype=torch.long).to(config.device))
    return initial_sample, original_images
