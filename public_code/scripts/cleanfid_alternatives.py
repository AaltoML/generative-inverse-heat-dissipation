"""This file contains mainly code from the clean-fid official 
repository (https://github.com/GaParmar/clean-fid/tree/main/cleanfid), 
but edited a bit to suit our needs"""

import string
from cleanfid.downloads_helper import check_download_url
import io
import pickle
import os
import random
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from scipy import linalg
import zipfile
import cleanfid
from cleanfid.utils import *
from cleanfid.features import build_feature_extractor  # , get_reference_statistics
from cleanfid.resize import *
import logging
import lmdb

"""
Cache a custom dataset statistics file
"""


def make_custom_stats(name, fdir, num=None, mode="clean",
                      num_workers=0, batch_size=64, device=torch.device("cuda"),
                      resolution=None):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    if resolution == None:  # THIS LINE ADDED
        split, res = "custom", "na"
    else:
        split, res = "custom", "{}".format(resolution)
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)

    feat_model = build_feature_extractor(mode, device)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images

    # custom_function_resize resizes the images to given resolution before resizing to (299,299)
    if resolution != None:
        custom_image_tranform = make_resizer(
            "PIL", False, "bicubic", (resolution, resolution))
    else:
        custom_image_tranform = None
    np_feats = get_folder_features(fdir, feat_model, num_workers=num_workers, num=num,
                                   batch_size=batch_size, device=device,
                                   mode=mode, description=f"custom stats: {fbname} : ",
                                   custom_image_tranform=custom_image_tranform)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
    # KID stats
    outf = os.path.join(stats_folder, f"{name}_{mode}_{split}_{res}_kid.npz")
    print(f"saving custom KID stats to {outf}")
    np.savez_compressed(outf, feats=np_feats)


"""
Compute the inception features for a folder of image files
"""


def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None):
    # get all relevant files in the dataset
    lmdbdata = False
    if ".zip" in fdir:
        files = list(set(zipfile.ZipFile(fdir).namelist()))
        # remove the non-image files inside the zip
        files = [x for x in files if os.path.splitext(x)[1].lower()[
            1:] in EXTENSIONS]
    elif "data.mdb" in os.listdir(fdir):
        lmdbdata = True
        files = []
    else:
        logging.info("Starting to gather files!")
        files = sorted([file for ext in EXTENSIONS
                        for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
        logging.info("... Done!")
    #logging.info("Folder: {}. Files: {}. Extensions: {}".format(fdir, files, EXTENSIONS))

    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device, mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  custom_image_tranform=custom_image_tranform,
                                  description=description, verbose=verbose,
                                  lmdbdata=lmdbdata, fdir=fdir)
    return np_feats


"""
Compute the inception features for a list of files
"""


def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None,
                       description="", verbose=True,
                       custom_image_tranform=None,
                       lmdbdata=False, fdir=None  # The last ones needed with lmdbdata
                       ):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)

    # wrap the images in a dataloader for parallelizing the resize operation
    if not lmdbdata:
        dataset = ResizeDataset(l_files, mode=mode)
    else:
        dataset = LMDBResizeDataset(mode=mode, fdir=fdir)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a batch of images
"""


def get_batch_features(batch, model, device):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


class LMDBResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores
    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    For lmdb format databases!
    """

    def __init__(self, mode, size=(299, 299), fdir=None):
        # This should be replaced with lmdb data
        #self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x

        # From torchvision.datasets.lsun.LSUNClass
        self.env = lmdb.open(fdir, max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + \
            "".join(c for c in fdir if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(
                    keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # From torchvision.datasets.lsun.LSUNClass
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[i])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img_np = np.array(Image.open(buf).convert("RGB"))

        # Rest is from clean-fid

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


"""
Load precomputed reference statistics for commonly used datasets
"""


def get_reference_statistics(name, res, mode="clean", seed=0, split="test", metric="FID"):
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    # if split == "custom":
    #    res = "na"
    if metric == "FID":
        rel_path = (f"{name}_{mode}_{split}_{res}.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    elif metric == "KID":
        rel_path = (f"{name}_{mode}_{split}_{res}_kid.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        return stats["feats"]


"""
custom_image_tranform:
    function that takes an np_array image as input [0,255] and 
    applies a custom transform such as cropping
"""


def compute_fid(fdir1=None, fdir2=None, gen=None,
                mode="clean", num_workers=12, batch_size=32,
                device=torch.device("cuda"), dataset_name="FFHQ",
                dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512,
                custom_feat_mode=None, verbose=True, custom_image_tranform=None):
    # build the feature extractor based on the mode
    if custom_feat_mode is None:
        feat_model = build_feature_extractor(mode, device)
    else:
        feat_model = custom_feat_mode

    # compute fid for a generator, using reference statistics
    if gen is not None:
        if verbose:
            logging.info(
                f"compute FID of a model with {dataset_name}-{dataset_res} statistics")
        score = fid_model(gen, dataset_name, dataset_res, dataset_split,
                          model=feat_model, z_dim=z_dim, num_gen=num_gen,
                          mode=mode, num_workers=num_workers, batch_size=batch_size,
                          device=device, verbose=verbose)
        return score

    # compute fid for a generator, using images in fdir2
    elif gen is not None and fdir2 is not None:
        if verbose:
            print(f"compute FID of a model, using references in fdir2")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                        batch_size=batch_size, device=device, mode=mode,
                                        description=f"FID {fbname2} : ", verbose=verbose,
                                        custom_image_tranform=custom_image_tranform)
        mu2 = np.mean(np_feats2, axis=0)
        sigma2 = np.cov(np_feats2, rowvar=False)
        # Generate test features
        np_feats = get_model_features(gen, feat_model, mode=mode,
                                      z_dim=z_dim, num_gen=num_gen,
                                      batch_size=batch_size, device=device, verbose=verbose)

        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        fid = frechet_distance(mu, sigma, mu2, sigma2)
        return fid

    else:
        raise ValueError(
            "invalid combination of directories and models entered")


"""
Computes the FID score between the two given folders
"""


def compare_folders(fdir1, fdir2, feat_model, mode, num_workers=0,
                    batch_size=8, device=torch.device("cuda"), verbose=True,
                    custom_image_tranform=None):
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname1} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_tranform)
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_tranform)
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


"""
Computes the FID score for a generator model for a specific dataset
and a specific resolution
"""


def fid_model(G, dataset_name, dataset_res, dataset_split,
              model=None, z_dim=512, num_gen=50_000,
              mode="clean", num_workers=0, batch_size=128,
              device=torch.device("cuda"), verbose=True):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    logging.info("dataset_res: {}".format(dataset_res))
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 mode=mode, seed=0, split=dataset_split,)

    # Generate test features
    np_feats = get_model_features(G, model, mode=mode,
                                  z_dim=z_dim, num_gen=num_gen,
                                  batch_size=batch_size, device=device, verbose=verbose)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


def get_model_features(G, model, mode="clean", z_dim=512,
                       num_gen=50_000, batch_size=128,
                       device=torch.device("cuda"), desc="FID model: ", verbose=True, return_z=False):
    fn_resize = build_resizer(mode)
    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    latents = []
    if verbose:
        pbar = tqdm(range(num_iters), desc=desc)
    else:
        pbar = range(num_iters)
    for idx in pbar:
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            if return_z:
                latents.append(z_batch)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                resized_batch = torch.zeros(batch_size, 3, 299, 299)
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    img_resize = fn_resize(img_np)
                    resized_batch[idx] = torch.tensor(
                        img_resize.transpose((2, 0, 1)))
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    if return_z:
        latents = torch.cat(latents, 0)
        return np_feats, latents
    return np_feats
