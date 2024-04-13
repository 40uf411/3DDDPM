from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, mask=None, pores_in_mask=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

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
        generate_mask=mask,
        pores_in_mask=pores_in_mask
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1, generate_mask=None, pores_in_mask=False):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.generate_mask = generate_mask
        self.pores_in_mask = pores_in_mask

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        
        # if it is a numpy array, just load it
        if path.endswith(".npy"):
            arr = np.load(path, allow_pickle=True)
            arr = arr.astype(np.float32)
            # Given that the data is 3D, if it the channel dimension is not set then add it (i.e. (50, 150, 150) -> (1, 50, 150, 150)))
            if len(arr.shape) == 3:
                arr = np.expand_dims(arr, 0)
            out_dict = {}
            if self.local_classes is not None:
                out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            if self.generate_mask is None:
                # generate a mask mask of the same size as the image with zeros
                mask = np.zeros(arr.shape, dtype=int)
            elif self.generate_mask == "random":
                mask = np.random.uniform(size=arr.shape) < 0.5
                mask = mask.astype(int)
            elif self.generate_mask == "slices":
                mask = np.zeros_like(arr)
                num_slices = np.random.randint(1, np.min(arr.shape[1:]))
                for _ in range(num_slices):
                    axis = np.random.randint(0, 3)
                    if axis == 0:
                        mask[0, np.random.randint(0, arr.shape[axis]), :, :] = 1
                    elif axis == 1:
                        mask[0, :, np.random.randint(0, arr.shape[axis]), :] = 1
                    elif axis == 2:
                        mask[0, :, :, np.random.randint(0, arr.shape[axis])] = 1
            elif self.generate_mask == "outer_contour":
                mask = np.zeros_like(arr)
                mask[0, 0, :, :] = 1
                mask[0, -1, :, :] = 1
                mask[0, :, 0, :] = 1
                mask[0, :, -1, :] = 1
                mask[0, :, :, 0] = 1
                mask[0, :, :, -1] = 1
            elif self.generate_mask.startswith("center_slice"):
                mask = np.zeros_like(arr)
                if self.generate_mask == "center_slice_y":
                    mask[0, :, arr.shape[2]//2, :] = 1
                elif self.generate_mask == "center_slice_z":
                    mask[0, :, :, arr.shape[3]//2] = 1
                else:
                    mask[0, arr.shape[1]//2, :, :] = 1
            else:
                raise np.random.uniform(size=(1))
            #ValueError(f"Unknown value for return_mask: {self.return_mask}")
            
            mask_2 = np.zeros_like(mask)
            # Get the indices where mask_1 is equal to 1
            indices = np.argwhere(mask == 1)
            # Set the corresponding indices in mask_2 to the values from 3D_image
            mask_2[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = arr[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
            if not self.pores_in_mask:
                mask_2[mask_2 == -1] = 0
            return arr, out_dict

        
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

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), np.zeros_like(arr), out_dict
