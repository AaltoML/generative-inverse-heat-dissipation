from ml_collections.config_flags import config_flags
from absl import app
from scripts import cleanfid_alternatives
from scripts import utils
import torch
from absl import flags
from scripts import datasets
from model_code import utils as mutils
from scripts import sampling
import os
import logging
import sys
FLAGS = flags.FLAGS

# Parameters related to i/o for loading model
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_integer("checkpoint", None,
                     "Checkpoint number to use for custom sampling")

# Sampling-related parameters
flags.DEFINE_float(
    "delta", 0.01, "The standard deviation of noise to add at each step with predicted reverse blur")
flags.DEFINE_integer(
    "batch_size", None, "Batch size of sampled images. Defaults to the training batch size")
flags.DEFINE_integer("skip_noise_starting_from", None,
                     "Sampling level after which to stop adding noise in the process")

# Cleanfid parameters
flags.DEFINE_string("dataset_name_cleanfid", None,
                    "Data set name, according to clean-fid naming convention")
flags.DEFINE_string("cleanfid_mode", "clean",
                    "clean-fid mode: clean or tensorflow-legacy")
flags.DEFINE_string("dataset_split", None,
                    "Dataset split, according to clean-fid naming convention. 'custom' if custom stats")

# Save custom cleanfid-stats
flags.DEFINE_bool("save_stats", False,
                  "Whether to use the save stats function")
flags.DEFINE_string("custom_name", None,
                    "Name of the new custom data set stats")
flags.DEFINE_string("dataset_path", None, "Path where we have the data set")

# Other parameters
flags.DEFINE_integer(
    "num_gen", 50000, "Number of samples generated for FID score")
flags.DEFINE_string(
    "mode", "fid", "Which thing to evaluate (fid/elbo/kid/inception)")

# Parameters related to saving the experiment results
flags.DEFINE_string("dataset_name", None,
                    "dataset name according to our convention (saves the results to the corresponding runs/dataset_name/)")
flags.DEFINE_string("experiment_name", None,
                    "Name of the evaluation experiment, e.g. sweep over some parameter")
flags.DEFINE_string(
    "param_name", "", "Within the given experiment, what makes this run different?")


def main(argv):
    if FLAGS.save_stats:
        save_stats(FLAGS.custom_name, FLAGS.dataset_path,
                   FLAGS.cleanfid_mode, FLAGS.image_size)
    elif FLAGS.mode == 'elbo':
        calculate_ELBO(FLAGS.config, FLAGS.dataset_name, FLAGS.experiment_name, FLAGS.param_name,
                       FLAGS.workdir, FLAGS.checkpoint, FLAGS.delta)
    else:
        calculate_fid(FLAGS.config, FLAGS.dataset_name, FLAGS.experiment_name, FLAGS.param_name,
                      FLAGS.workdir, FLAGS.checkpoint, FLAGS.delta,
                      FLAGS.dataset_name_cleanfid, FLAGS.dataset_split, FLAGS.num_gen, FLAGS.cleanfid_mode,
                      FLAGS.batch_size)

def calculate_fid(config, dataset_name, experiment_name, param_name,
                  workdir, checkpoint, delta, dataset_name_cleanfid,
                  dataset_split, num_gen=50000, mode='clean', batch_size=None):
    # dataset_name: e.g. cifar10, mnist
    # experiment name: e.g. scores_sweep_training_steps_{model name}
    # param_name: what is the param that is changed around in this run?

    image_size = config.data.image_size
    K = config.model.K
    assert batch_size != None

    start_level = 0

    logging.info("Running on {}".format(config.device))
    if checkpoint > 0:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(
            config, checkpoint_dir, checkpoint)
    else:
        checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)
    model.eval()

    # Get the forward process definition
    logging.info("Getting filter scales")
    scales = config.model.blur_schedule
    logging.info("Creating the filter array...")
    forward_process_module = mutils.create_forward_process_from_sigmas(
        config, scales, config.device)
    logging.info("Done")

    # Get the data
    trainloader, testloader = datasets.get_dataset(config,
                                                   uniform_dequantization=config.data.uniform_dequantization,
                                                   train_batch_size=batch_size, num_workers=2)

    train_iter = iter(trainloader)
    batch_size = next(train_iter)[0].shape[0]

    logging.info(config)

    # Wrapper for generating random data:
    def generate_samples(dummy_argument, train_iter=train_iter,
                         trainloader=trainloader):
        try:
            batch = next(train_iter)[0].to(config.device).float()
            if batch.shape[0] < batch_size:
                train_iter = iter(trainloader)
                batch = next(train_iter)[0].to(config.device).float()
        except StopIteration:
            train_iter = iter(trainloader)
            batch = next(train_iter)[0].to(config.device).float()

        batch = forward_process_module(
            batch, K*torch.ones(batch.shape[0], dtype=torch.long, device=config.device))
        # add the initial noise
        batch = batch + torch.randn_like(batch) * delta
        sampling_fn = sampling.get_sampling_fn_inverse_heat(config,
                                                            batch, intermediate_sample_indices=None, delta=delta, device=config.device)

        sample, _, _ = sampling_fn(model)
        sample = torch.clip(sample*255., 0, 255).to(torch.uint8)
        return sample

    logging.info("{}_{}_{}_{}.npz".format(
        dataset_name_cleanfid, mode, dataset_split, image_size))
    fid_score = cleanfid_alternatives.compute_fid(gen=generate_samples, dataset_name=dataset_name_cleanfid,
                                                  dataset_res=image_size, num_gen=num_gen, dataset_split=dataset_split,
                                                  mode=mode, batch_size=batch_size, device=config.device)
    logging.info("FID score: {:.3f}".format(fid_score))

    utils.append_to_log_file(dataset_name, experiment_name,
                             param_name, "FID_{}_{}".format(mode, num_gen), fid_score)
    utils.append_to_dict(dataset_name, experiment_name,
                         param_name, "FID_{}_{}".format(mode, num_gen), fid_score)


def save_stats(custom_name, dataset_path, cleanfid_mode, resolution):
    # Doesn't require data set split, is set to custom in clean-fid
    logging.info("Calculating stats for FID scores...")
    logging.info("cleanfid mode: {}".format(cleanfid_mode))
    logging.info("Folder: {}".format(dataset_path))
    cleanfid_alternatives.make_custom_stats(custom_name,
                                            dataset_path, mode=cleanfid_mode,
                                            resolution=resolution)
    logging.info("Done!")


def calculate_ELBO(config, dataset_name, experiment_name, param_name,
                   workdir, checkpoint, delta):
    # dataset_name: e.g. cifar10, mnist
    # experiment name: e.g. scores_sweep_training_steps_{model name}
    # param_name: what is the param that is changed around in this run?

    image_size = config.data.image_size
    logging.info("Running on {}".format(config.device))
    if checkpoint > 0:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(
            config, checkpoint_dir, checkpoint)
    else:
        checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)
    model.eval()

    # Get the forward process definition
    scales = config.model.blur_schedule
    logging.info("Creating the forward process...")
    blur_module = mutils.create_forward_process_from_sigmas(
        config, scales, config.device)
    logging.info("Done")

    # Get the data
    trainloader, testloader = datasets.get_dataset(config,
                                                   uniform_dequantization=True, train_batch_size=10000, eval_batch_size=128)
    train_size = len(trainloader.dataset)
    test_size = len(testloader.dataset)

    sigma = config.model.sigma
    delta = delta
    L_K_upbound, L_others, mse_losses = mutils.neg_ELBO(config, trainloader, testloader, blur_module, sigma, delta, image_size,
                                                        train_size, test_size, model, config.device, num_epochs=100)

    return_val = {"total": L_K_upbound.mean()+sum(L_others), "L_K_upbound": L_K_upbound.mean(), "L_others": L_others,
                  "mse_losses": mse_losses}
    utils.append_to_log_file(
        dataset_name, experiment_name, param_name, "ELBO", return_val)
    utils.append_to_dict(dataset_name, experiment_name,
                         param_name, "ELBO", return_val)


if __name__ == "__main__":
    app.run(main)
