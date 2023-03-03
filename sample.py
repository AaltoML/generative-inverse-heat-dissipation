import os
from pathlib import Path
import logging
from scripts import sampling
from model_code import utils as mutils
import torch
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("checkpoint", None,
                     "Checkpoint number to use for custom sampling")
flags.mark_flags_as_required(["workdir", "config", "checkpoint"])
flags.DEFINE_integer("save_sample_freq", 1,
                     "How often to save samples for output videos?")
flags.DEFINE_float(
    "delta", 0.01, "The standard deviation of noise to add at each step with predicted reverse blur")
flags.DEFINE_integer(
    "batch_size", None, "Batch size of sampled images. Defaults to the training batch size")
flags.DEFINE_bool("same_init", False,
                  "Whether to initialize all samples at the same image")
flags.DEFINE_bool("share_noise", False,
                  "Whether to use the same noises for each image in the generated batch")
flags.DEFINE_integer(
    "num_points", 10, "Default amount of points for sweeping the input from one place to another")
flags.DEFINE_float("final_noise", None,
                   "How much should the noise at the end be? Linear interpolation from noise_amount ot this. If none, use noise_amount")
flags.DEFINE_bool("interpolate", False, "Whether to do interpolation")
flags.DEFINE_integer(
    "number", None, "add a number suffix to generated sample in interpolate")


def main(argv):
    if FLAGS.interpolate:
        sample_interpolate(FLAGS.config, FLAGS.workdir, FLAGS.checkpoint,
                           FLAGS.delta, FLAGS.num_points, FLAGS.number)
    else:
        sample(FLAGS.config, FLAGS.workdir, FLAGS.checkpoint, FLAGS.save_sample_freq, FLAGS.delta,
               FLAGS.batch_size, FLAGS.share_noise, FLAGS.same_init)


def sample(config, workdir, checkpoint, save_sample_freq=1,
           delta=None, batch_size=None, share_noise=False, same_init=False):

    if batch_size == None:
        batch_size = config.training.batch_size

    if checkpoint > 0:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(
            config, checkpoint_dir, checkpoint)
    else:  # Checkpoint means the latest checkpoint
        checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)

    model_fn = mutils.get_model_fn(model, train=False)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    logging.info("Running on {}".format(config.device))

    logging.info("Creating the forward process...")
    scales = config.model.blur_schedule
    heat_forward_module = mutils.create_forward_process_from_sigmas(
        config, scales, config.device)
    logging.info("Done")
    initial_sample, original_images = sampling.get_initial_sample(
        config, heat_forward_module, delta, batch_size)
    if same_init:
        initial_sample = torch.cat(batch_size*[initial_sample[0][None]], 0)
        original_images = torch.cat(batch_size*[original_images[0][None]], 0)
    initial_sample, original_images = initial_sample[:
                                                     batch_size], original_images[:batch_size]
    sampling_shape = initial_sample.shape

    intermediate_sample_indices = list(
        range(0, config.model.K+1, save_sample_freq))
    sample_dir = os.path.join(workdir, "additional_samples")
    this_sample_dir = os.path.join(
        sample_dir, "checkpoint_{}".format(checkpoint))

    # Get smapling function and save directory
    sampling_fn = sampling.get_sampling_fn_inverse_heat(config, initial_sample,
                                                        intermediate_sample_indices, delta, config.device, share_noise=share_noise)
    this_sample_dir = os.path.join(this_sample_dir, "delta_{}".format(delta))
    if same_init:
        this_sample_dir += "_same_init"
    if share_noise:
        this_sample_dir += "_share_noise"

    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Do sampling")
    sample, n, intermediate_samples = sampling_fn(model_fn)

    # Save results
    utils.save_tensor_list(this_sample_dir, intermediate_samples, "samples.np")
    utils.save_tensor(this_sample_dir, sample, "final.np")
    utils.save_png(this_sample_dir, sample, "final.png")
    utils.save_png(this_sample_dir, initial_sample, "init.png")
    utils.save_gif(this_sample_dir, intermediate_samples)
    utils.save_video(this_sample_dir, intermediate_samples)


def sample_interpolate(config, workdir, checkpoint,
                       delta, num_points, number):
    # The interpolation function returns only one interpolation between two random points
    # -> batch_size = 2
    batch_size = 2
    if checkpoint > 0:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(
            config, checkpoint_dir, checkpoint)
    else:
        checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)

    model_fn = mutils.get_model_fn(model, train=False)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    logging.info("Running on {}".format(config.device))
    logging.info("Creating the forward process...")
    scales = config.model.blur_schedule
    heat_forward_module = mutils.create_forward_process_from_sigmas(
        config, scales, config.device)
    logging.info("Done")
    initial_sample, original_images = sampling.get_initial_sample(
        config, heat_forward_module, delta, batch_size)
    initial_sample = initial_sample[:batch_size]

    # Directory name for saving results
    sample_dir = os.path.join(workdir, "additional_samples")
    this_sample_dir = os.path.join(
        sample_dir, "checkpoint_{}".format(checkpoint))

    # Get the sampling function
    sampling_fn, init_input = sampling.get_sampling_fn_inverse_heat_interpolate(
        config, initial_sample,
        delta, device=config.device, num_points=num_points)
    this_sample_dir = os.path.join(
        this_sample_dir, "interpolate_delta_{}".format(delta))
    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
    utils.save_png(this_sample_dir, init_input,
                   "init_input.png", nrow=num_points)

    x_sweep = sampling_fn(model_fn)
    logging.info("Sampling done!")

    logging.info("Saving results...")
    if number == None:
        utils.save_png(this_sample_dir, x_sweep[0:1], "base_sample.png")
        utils.save_png(this_sample_dir, x_sweep,
                       "interpolation.png", nrow=num_points)
        print(x_sweep[-1:].repeat(2, 1, 1, 1))
        utils.save_video(this_sample_dir,
                         torch.cat([x_sweep, x_sweep[-1:].repeat(10, 1, 1, 1), reversed(x_sweep), x_sweep[:1].repeat(10, 1, 1, 1)]))
    else:
        utils.save_png(this_sample_dir,
                       x_sweep[0:1], "base_sample_{}.png".format(number))
        utils.save_png(this_sample_dir, x_sweep,
                       "interpolation_{}.png".format(number), nrow=num_points)
        utils.save_video(this_sample_dir,
                         torch.cat([x_sweep, x_sweep[-1:].repeat(10, 1, 1, 1), reversed(x_sweep), x_sweep[:1].repeat(10, 1, 1, 1)]))
    logging.info("Done!")


if __name__ == "__main__":
    app.run(main)
