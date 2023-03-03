"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from model_code import utils as mutils
import torch.distributions


def get_optimizer(config, params):
    """Returns an optimizer object based on `config`.
    Copied from https://github.com/yang-song/score_sde_pytorch"""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`.
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if config.optim.automatic_mp:
        def optimize_fn(optimizer, params, step, scaler, lr=config.optim.lr,
                        warmup=config.optim.warmup,
                        grad_clip=config.optim.grad_clip):
            """Optimizes with warmup and gradient clipping (disabled if negative).
            Before that, unscales the gradients to the regular range from the 
            scaled values for automatic mixed precision"""
            scaler.unscale_(optimizer)
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            # Since grads already scaled, this just takes care of possible NaN values
            scaler.step(optimizer)
            scaler.update()
    else:
        def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                        warmup=config.optim.warmup,
                        grad_clip=config.optim.grad_clip):
            """Optimizes with warmup and gradient clipping (disabled if negative)."""
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
    return optimize_fn


def get_label_sampling_function(K):
    return lambda batch_size, device: torch.randint(1, K, (batch_size,), device=device)


def get_inverse_heat_loss_fn(config, train, scales, device, heat_forward_module):

    sigma = config.model.sigma
    label_sampling_fn = get_label_sampling_function(config.model.K)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(
            model, train=train)  # get train/eval model
        fwd_steps = label_sampling_fn(batch.shape[0], batch.device)
        blurred_batch = heat_forward_module(batch, fwd_steps).float()
        less_blurred_batch = heat_forward_module(batch, fwd_steps-1).float()
        noise = torch.randn_like(blurred_batch) * sigma
        perturbed_data = noise + blurred_batch
        diff = model_fn(perturbed_data, fwd_steps)
        prediction = perturbed_data + diff
        losses = (less_blurred_batch - prediction)**2
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss, losses, fwd_steps

    return loss_fn


def get_step_fn(train, scales, config, optimize_fn=None,
                heat_forward_module=None, device=None):
    """A wrapper for loss functions in training or evaluation
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if device == None:
        device = config.device

    loss_fn = get_inverse_heat_loss_fn(config, train,
                                       scales, device, heat_forward_module=heat_forward_module)

    # For automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    def step_fn(state, batch):
        """Running one step of training or evaluation.
        Returns:
                loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            if config.optim.automatic_mp:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, losses_batch, fwd_steps_batch = loss_fn(model, batch)
                    # amp not recommended in backward pass, but had issues getting this to work without it
                    # Followed https://github.com/pytorch/pytorch/issues/37730
                    scaler.scale(loss).backward()
                scaler.scale(losses_batch)
                optimize_fn(optimizer, model.parameters(), step=state['step'],
                            scaler=scaler)
                state['step'] += 1
                state['ema'].update(model.parameters())
            else:
                optimizer.zero_grad()
                loss, losses_batch, fwd_steps_batch = loss_fn(model, batch)
                loss.backward()
                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['step'] += 1
                state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss, losses_batch, fwd_steps_batch = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss, losses_batch, fwd_steps_batch

    return step_fn
