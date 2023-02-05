import json
import math
import random
from os import listdir
from os.path import isfile, join
import pickle
import bisect

import clearml as clearml
from pathlib import Path
from PIL import Image

import torchvision
import torchvision.transforms.functional as F
import tqdm
from torch.utils.data import Dataset
import torch
import numpy as np

from dataset.utils import show_dataset


class AugmentedPickingDataset(Dataset):
    def __init__(self, backbone: Dataset, overwrite_num_rotations=None):
        self.backbone = backbone
        self.overwrite_num_rotations = overwrite_num_rotations

    def __len__(self):
        return len(self.backbone)

    def __getitem__(self, item):
        result = self.backbone[item]
        t_range = 200
        dt_range = 2
        row, col = -1, -1
        idx = result["grasping_index"]
        h = result["heightmap"]
        n = self.overwrite_num_rotations if self.overwrite_num_rotations is not None else result["num_rotations"]
        if "num_rotations" not in result: result["num_rotations"] = self.overwrite_num_rotations

        def im_to_cart(row, col):
            return col - h.shape[-1] / 2, -row + h.shape[-1] / 2

        def cart_to_im(x, y):
            return math.floor(h.shape[-1] / 2 - y), math.floor(x + h.shape[-1] / 2)

        def rotate_2dvector(x, y, theta):
            res = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]
            ).T.dot((x, y))
            return res[0], res[1]

        while not (0 <= row < h.shape[-2] and 0 <= col < h.shape[-1]):
            theta_idx = random.randint(0, n - 1)
            t = [random.randint(-t_range, t_range), random.randint(-t_range, t_range)]
            x, y = im_to_cart(idx["row"], idx["col"])
            x, y = rotate_2dvector(x, y, -theta_idx / n * math.pi)
            row, col = cart_to_im(x, y)
            row, col = row + t[1], col + t[0]

        idx["row"], idx["col"] = row, col
        idx["angle_index"] = (idx["angle_index"] * (n // result["num_rotations"]) + theta_idx) % n

        scale = np.random.uniform(0.98, 1.02)
        shear = list(np.random.uniform(-1, 1, 2))
        angle_noise = np.random.uniform(-0.5, 0.5)

        result["pure_heightmap"] = F.affine(h, -theta_idx / n * 180, [t[0], t[1]], scale=1., shear=[0., 0.])

        dt = [random.randint(-dt_range, dt_range), random.randint(-dt_range, dt_range)]
        h = F.affine(h, -theta_idx / n * 180 + angle_noise, [t[0] + dt[0], t[1] + dt[1]], scale, shear)

        # Gaussian noise
        noise = np.random.normal(0, 0.01, h.shape)
        h += noise

        result["heightmap"] = h
        result["grasping_index"] = idx

        return result


class ExtendedPickingDataset(Dataset):
    def __init__(self, root_dir, trajectory_limit=None):
        self.root_dir = root_dir
        self.files = [f for f in listdir(root_dir)[:trajectory_limit] if isfile(join(root_dir, f))]
        self.lengths = [len(self.load_pickle(f)) for f in tqdm.tqdm(self.files)]
        self.cum_lengths = \
            list(np.cumsum([len(self.load_pickle(f)) for f in self.files]))

    def load_pickle(self, fname):
        with open(join(self.root_dir, fname), 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, fname, result):
        with open(join(self.root_dir, fname), 'wb') as f:
            pickle.dump(result, f)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        traj_idx = bisect.bisect_left(self.cum_lengths, idx)
        fname = self.files[traj_idx]
        traj = self.load_pickle(fname)
        result = traj[idx - self.cum_lengths[traj_idx]]
        result["heightmap"] = result["heightmap"].type(torch.FloatTensor)

        return {
            "heightmap": result["heightmap"],
            "successful": result["successful"],
            "grasping_index": result["grasping_index"],
        }

    def normalize_picking_dataset(self, normalization):
        for filename in self.files:
            pkl = self.load_pickle(filename)
            for x in pkl:
                x["heightmap"] = normalization(x["heightmap"])
            self.save_pickle(filename, pkl)

    def convert_to_picking_dataset(self, directory):
        for i in range(len(self)):
            x = self[i]
            with open(f"{directory}/{i}.pkl", "wb") as f:
                pickle.dump({
                    "heightmap": x["heightmap"],
                    "successful": x["successful"],
                    "grasping_index": x["grasping_index"],
                }, f)



class PickingDataset(torch.utils.data.Dataset):
    mean = [0.5056109428405762, 0.45094895362854004, 0.4510827958583832, 0.00021378498058766127]
    std = [0.06538673490285873, 0.07051316648721695, 0.07949845492839813, 0.0017423119861632586]

    normalize = torchvision.transforms.Normalize(mean, std, inplace=False)

    def __init__(self, path, transforms=None):
        self.path = Path(path)
        self.transforms = transforms
        self.dates = [p.name[4:-4] for p in self.path.iterdir() if 'img' in p.name]
        self.img_names = [f"img_{date}.png" for date in self.dates]
        self.depth_names = [f"depth_{date}.npy" for date in self.dates]
        self.desc_names = [f"desc_{date}.json" for date in self.dates]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        img_path = self.path / self.img_names[idx]
        depth_path = self.path / self.depth_names[idx]
        desc_path = self.path / self.desc_names[idx]

        img = torchvision.transforms.ToTensor()(Image.open(img_path))
        depth = torch.tensor(np.load(depth_path)).unsqueeze(0)
        raw_heightmap = torch.cat((img, depth), dim=0)

        with open(desc_path, 'r') as fdesc:
            description = json.load(fdesc)

        return {
            "heightmap": raw_heightmap,
            "successful": description["successful"],
            "grasping_index": description["grasping_index"],
            "num_rotations": description["num_rotations"]
        }



if __name__ == "__main__":
    root_dir = "dataset/dataset-cables"
    dataset = ExtendedPickingDataset(root_dir)
    show_dataset(dataset, lambda x: x["heightmap"][3])
    show_dataset(dataset, lambda x: x["heightmap"][:3].permute((1, 2, 0)))
