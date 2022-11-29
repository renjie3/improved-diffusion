from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, output_index=False, output_class=False, mode="train", poisoned=False, poisoned_path='', hidden_class=0, num_input_channels=3, num_workers=1,
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
    if output_class:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        adv_output_classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        num_input_channels=num_input_channels,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        output_index=output_index,
        output_classes=adv_output_classes,
        poisoned=poisoned,
        poisoned_path=poisoned_path,
        hidden_class=hidden_class,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader


def load_adv_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, output_index=False, mode="train", adv_noise_num=5000, output_class=False, single_target_image_id=10000, num_input_channels=3, num_workers=1, poison_mode="gradient_matching", source_dir=None, source_class=0, one_class_image_num=5000, source_clean_dir=None,
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
    if output_class:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        adv_output_classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        num_input_channels=num_input_channels,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        output_index=output_index,
        one_class_image_num=one_class_image_num,
        output_class_flag=output_class,
        output_classes=adv_output_classes,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False # check it
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
        )
    if poison_mode=="gradient_matching":
        adv_noise = np.zeros([len(dataset), num_input_channels, image_size, image_size])
        if not source_dir:
            raise ValueError("unspecified data directory")
        all_source_files = _list_image_files_recursively(source_dir)
        classes = None
        adv_output_classes = None
        source_dataset = ImageDataset(
            image_size,
            all_source_files,
            num_input_channels=num_input_channels,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            output_index=output_index,
            # one_class_image_num=adv_noise_num,
            output_class_flag=False,
            output_classes=adv_output_classes,
        )
        source_loader = DataLoader(
                source_dataset, batch_size=batch_size // 10, shuffle=False, num_workers=num_workers, drop_last=False # check it
            )

        if source_clean_dir != None:
            if not source_clean_dir:
                raise ValueError("unspecified data directory")
            all_source_clean_files = _list_image_files_recursively(source_clean_dir)
            classes = None
            adv_output_classes = None
            source_clean_dataset = ImageDataset(
                image_size,
                all_source_clean_files,
                num_input_channels=num_input_channels,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                output_index=output_index,
                # one_class_image_num=adv_noise_num,
                output_class_flag=False,
                output_classes=adv_output_classes,
            )
            source_clean_loader = DataLoader(
                    source_clean_dataset, batch_size=batch_size // 10, shuffle=False, num_workers=num_workers, drop_last=False # check it
                )
        else:
            raise("source_clean_dir == None Not emplemented.")
    else:
        adv_noise = np.zeros([adv_noise_num, num_input_channels, image_size, image_size])
    if mode == "adv":
        if poison_mode=="gradient_matching":
            return loader, source_loader, source_clean_loader, adv_noise
        else:
            target_image, target_dict = dataset[single_target_image_id]
            return loader, adv_noise, target_image
    else:
        raise("Only mode == adv uses load_adv_data.")


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
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1, output_index=False, one_class_image_num=1, output_class_flag=False, output_classes=None, poisoned=False, poisoned_path=None, hidden_class=0, num_input_channels=3):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.output_index = output_index
        self.one_class_image_num = one_class_image_num
        self.output_class_flag = output_class_flag
        self.output_classes = output_classes
        self.hidden_class = hidden_class
        self.num_input_channels = num_input_channels
        self.poisoned=poisoned
        if self.poisoned:
            with open('{}.npy'.format(poisoned_path), 'rb') as f:
                self.perturb = np.load(f)
                print("Poison loaded!!! at {}".format(poisoned_path))

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # print(len(self.local_images))
        # input('check')
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

        if self.num_input_channels != 1:
            arr = np.array(pil_image.convert("RGB"))
        else:
            arr = np.expand_dims(np.array(pil_image), 2)
            # print(arr.shape)
            # input('check')
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1 # because the range is [-1,1], the perturbation should double to 0.0314 * 2. Clip also needs to be modified.
        arr = np.transpose(arr, [2, 0, 1])
        if self.poisoned:
            if self.output_classes[idx] == self.hidden_class:
                arr += self.perturb[idx % self.one_class_image_num]
                # input('check here')

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.output_index:
            # out_dict["idx"] = np.array(idx % self.one_class_image_num, dtype=np.int64)
            out_dict["idx"] = np.array(idx, dtype=np.int64)
        if self.output_class_flag:
            out_dict["output_classes"] = np.array(self.output_classes[idx], dtype=np.int64)
        return arr, out_dict
