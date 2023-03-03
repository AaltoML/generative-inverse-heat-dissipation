"""Some parts based on https://github.com/yang-song/score_sde_pytorch"""

from torch.utils.data import DataLoader, Dataset
import numpy as np
from mpi4py import MPI
import blobfile as bf
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, datasets
import torch
from PIL import Image


class UniformDequantize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return (torch.rand(sample.shape) + sample*255.)/256.


def get_dataset(config, uniform_dequantization=False, train_batch_size=None,
                eval_batch_size=None, num_workers=8):
    """
    Get Pytorch dataloaders for one of the following datasets:
    MNIST, CIFAR-10, LSUN-Church, FFHQ, AFHQ
    MNIST and CIFAR-10 are loaded through torchvision, others have to be
    downloaded separately to the data/ folder from the following sources:
    https://github.com/NVlabs/ffhq-dataset
    https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq
    https://github.com/fyu/lsun
    """

    transform = [transforms.Resize(config.data.image_size),
                 transforms.CenterCrop(config.data.image_size)]
    if config.data.random_flip:
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    if uniform_dequantization:
        transform.append(UniformDequantize())
    transform = transforms.Compose(transform)

    if not train_batch_size:
        train_batch_size = config.training.batch_size
    if not eval_batch_size:
        eval_batch_size = config.eval.batch_size

    if config.data.dataset == 'MNIST':
        training_data = datasets.MNIST(
            root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=transform)
    elif config.data.dataset == 'CIFAR10':
        training_data = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform)
    elif config.data.dataset == "lsun_church":
        training_data = datasets.LSUN(
            root="data/lsun", classes=['church_outdoor_train'], transform=transform)
        test_data = datasets.LSUN(
            root="data/lsun", classes=['church_outdoor_val'], transform=transform)
    elif config.data.dataset == 'FFHQ':
        trainloader = load_data(data_dir="data/ffhq-dataset/images1024x1024",
                                batch_size=train_batch_size, image_size=config.data.image_size,
                                random_flip=config.data.random_flip)
        testloader = load_data(data_dir="data/ffhq-dataset/images1024x1024",
                               batch_size=eval_batch_size, image_size=config.data.image_size,
                               random_flip=False)
        return trainloader, testloader
    elif config.data.dataset == 'AFHQ':
        trainloader = load_data(data_dir="data/afhq/train",
                                batch_size=train_batch_size, image_size=config.data.image_size,
                                random_flip=config.data.random_flip)
        testloader = load_data(data_dir="data/afhq/val",
                               batch_size=eval_batch_size, image_size=config.data.image_size,
                               random_flip=False)
        return trainloader, testloader
    else:
        raise ValueError

    # If we didn't use the load_data function that already created data loaders:
    trainloader = DataLoader(training_data, batch_size=train_batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=eval_batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    return trainloader, testloader


""" The following mostly pasted from the Improved Denoising Diffusion models github page:
		https://github.com/openai/improved-diffusion """


def load_data(
        *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
        random_flip=True
):
    """
    NOTE: Change to original function, returns the Pytorch dataloader, not a generator

    For a dataset, create a dataloader over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                                       label. If classes are not available and this is true, an
                                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1,
                 random_flip=True):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        # Random horizontal flip
        if self.random_flip:
            if np.random.rand() > 0.5:
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution,
                  crop_x: crop_x + self.resolution]
        # Changed here so that not centered at zero
        # arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.astype(np.float32) / 255

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
