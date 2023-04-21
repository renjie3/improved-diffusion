from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, output_index=False, output_class=False, mode="train", poisoned=False, poisoned_path='', hidden_class=0, num_workers=4, 
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
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, output_index=False, mode="train", adv_noise_num=5000, output_class=False, single_target_image_id=10000, num_workers=4, 
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
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        output_index=output_index,
        one_class_image_num=adv_noise_num,
        output_class_flag=output_class,
        output_classes=adv_output_classes,
        hidden_class=hidden_class,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False # check it
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
        )
    adv_noise = np.zeros([adv_noise_num, 3, image_size, image_size])
    if mode == "adv":
        target_image, target_dict = dataset[single_target_image_id]
        return loader, adv_noise, target_image
    else:
        raise("Only adv uses load_adv_data.")


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
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1, output_index=False, one_class_image_num=1, output_class_flag=False, output_classes=None, poisoned=False, poisoned_path=None, hidden_class=0):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.output_index = output_index
        self.one_class_image_num = one_class_image_num
        self.output_class_flag = output_class_flag
        self.output_classes = output_classes
        self.hidden_class = hidden_class
        self.poisoned=poisoned
        if self.poisoned:
            with open('{}.npy'.format(poisoned_path), 'rb') as f:
                self.perturb = np.load(f)

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

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1 # because the range is [-1,1], the perturbation should double to 0.0314 * 2. Clip also needs to be modified.
        arr = np.transpose(arr, [2, 0, 1])
        if self.poisoned:
            if self.output_classes[idx] == self.hidden_class:
                arr += self.perturb[idx]
                # input('check here')

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.output_index:
            out_dict["idx"] = np.array(idx, dtype=np.int64)
        if self.output_class_flag:
            out_dict["output_classes"] = np.array(self.output_classes[idx], dtype=np.int64)
        return arr, out_dict


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform, use_numpy_file=False):
        super().__init__()
        self.use_numpy_file = use_numpy_file
        self.transform = transform
        if self.use_numpy_file:
            # self.local_classes = pickle.load(image_paths.replace("uint8.npy", "label.pkl"))
            with open(image_paths.replace("uint8.npy", "label.pkl"),'rb') as out_data:
                self.local_classes = pickle.load(out_data)
            self.numpy_data = np.load(image_paths)
        else:
            self.local_images = image_paths
            class_names = [bf.basename(path).split("_")[0] for path in image_paths]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
            self.local_classes = classes

        # pngs = []
        # for path in image_paths:
        #     with bf.BlobFile(path, "rb") as f:
        #         pil_image = Image.open(f)
        #         pil_image.load()

        #     pil_image = pil_image.convert("RGB")
        #     arr = np.array(pil_image)
        #     pngs.append(arr)

        # pngs = np.stack(pngs, axis=0)
        # np.save('./datasets/cifar10_test_uint8.npy', pngs)

        # temp = [bf.basename(path).split("_")[1].split(".")[0] for path in image_paths]
        # # print(temp[:10])
        # # input("check")
        # with open('./datasets/cifar10_train_id.pkl','wb') as in_data:
        #     pickle.dump(temp,in_data,pickle.HIGHEST_PROTOCOL)
        # exit()

        # with open('cifar10_train_names.pkl','wb') as in_data:
        #     pickle.dump(self.local_classes,in_data,pickle.HIGHEST_PROTOCOL)

        # exit()

    def __len__(self):
        return len(self.local_classes)

    def __getitem__(self, idx):
        if self.use_numpy_file:
            pil_image = Image.fromarray(self.numpy_data[idx])
        else:
            path = self.local_images[idx]
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()

        pil_image = pil_image.convert("RGB")

        img = self.transform(pil_image)

        label = np.array(self.local_classes[idx], dtype=np.int64)
        return img, label, idx

class SimpleImageDatasetWithSelfWatermark(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.transform = transform
        with open(image_paths.replace("uint8.npy", "watermark_label.pkl").replace("cifar10_test", "cifar10_train"),'rb') as out_data:
            self.local_classes = pickle.load(out_data)

        self.numpy_data = np.load(image_paths).astype(np.float32)

        # self.test_numpy_data = np.load(image_paths.replace("cifar10_train", "cifar10_test")).astype(np.float32)

        # with open(image_paths.replace("cifar10_train", "cifar10_test").replace("uint8.npy", "label.pkl"),'rb') as out_data:
        #     self.test_local_classes = pickle.load(out_data)

        # watermark_id = np.random.permutation(self.test_numpy_data.shape[0])

        # watermark_numpy_data = []
        # watermark_label = []
        # for _id in watermark_id:
        #     watermark_numpy_data.append(self.train_numpy_data[_id])
        #     watermark_label.append(self.train_local_classes[_id])

        # watermark_numpy_data = np.stack(watermark_numpy_data, axis=0)

        # np.save('./datasets/cifar10_test_watermark_uint8.npy', watermark_numpy_data)

        # with open('cifar10_test_watermark_label.pkl','wb') as in_data:
        #     pickle.dump(watermark_label,in_data,pickle.HIGHEST_PROTOCOL)

        # print("check")
        # exit()

        self.watermark_numpy_data = np.load(image_paths.replace("uint8.npy", "watermark_uint8.npy").replace("cifar10_test", "cifar10_train")).astype(np.float32)
        # self.numpy_data += self.watermark_numpy_data / 255.0 * 16 - 8
        # self.numpy_data = np.clip(self.numpy_data, 0, 255).astype(np.uint8)

        # self.numpy_data = np.clip(self.watermark_numpy_data, 0, 255).astype(np.uint8)

        class_set = set()

        with open("./datasets/cifar10_train_id.pkl",'rb') as out_data:
            image_paths = pickle.load(out_data)
        for i in range(5000):
            new_idx = int(min(i // 10 * 50 // 10 * 10, 25000))
            class_set.add(new_idx)
            # print(new_idx)
            arr = self.numpy_data[i+5000] + self.numpy_data[25000 + new_idx] / 255.0 * 64 - 32
            # arr = self.numpy_data[25000 + new_idx]
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(arr)
            pil_image.convert('RGB').save("./datasets/unversal/5c500c/{:05d}.png".format(i))
            # input("check")
        # print(class_set)
        print(len(class_set))
        print(max(class_set))

        exit()


    def __len__(self):
        return len(self.local_classes)

    def __getitem__(self, idx):

        random_idx = np.random.randint(0, self.numpy_data.shape[0])

        new_idx = idx // 100 * 100

        arr = self.numpy_data[random_idx] + self.watermark_numpy_data[new_idx] / 255.0 * 64 - 32
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(arr)

        # print(idx)
        # pil_image.convert('RGB').save("test.png")
        # input("check")

        # pil_image = Image.fromarray(self.numpy_data[idx])

        pil_image = pil_image.convert("RGB")

        img = self.transform(pil_image)

        label = np.array(self.local_classes[new_idx], dtype=np.int64)
        return img, label, idx
