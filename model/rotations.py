import abc
import copy
from collections import OrderedDict

import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from dataset.pointwise import PickingDataset


class RotationsModule(nn.Module, abc.ABC):
    def __init__(self, backbone: nn.Module, num_rotations=8, max_angle=math.pi, padding=True, padding_noise=0., noise_normalization=True):
        super().__init__()
        self.backbone = backbone
        self.backbone.train()

        self.num_rotations = num_rotations
        self.max_angle = max_angle
        self.padding = padding
        self.padding_noise = padding_noise
        self.noise_normalization = noise_normalization

    @staticmethod
    def rotate_2dvector(x, y, theta):
        res = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]).T.dot((x, y))
        return res[0], res[1]

    def cuda(self, device=None):
        return AffordanceRotationsModule(self.backbone.cuda(device), num_rotations=self.num_rotations, max_angle=self.max_angle)

    # https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports autograd
    @staticmethod
    def rotation_matrix2d(theta):
        if len(theta.shape) == 0:
            theta = torch.tensor([theta])
        zero = torch.zeros_like(theta)
        row0 = torch.stack([torch.cos(theta), -torch.sin(theta), zero])
        row1 = torch.stack([torch.sin(theta), torch.cos(theta), zero])
        result = torch.stack([row0, row1])
        return result.permute(2, 0, 1)

    @staticmethod
    def rotate_image(x, theta, mode="nearest"):
        """ Rotates image counterclock-wise by theta in [0; 2 * pi). """
        if type(theta) == float:
            theta = torch.tensor(theta)
        m = AffordanceRotationsModule.rotation_matrix2d(theta.clone().detach())
        # m = torch.concat([m] * x.shape[0])
        grid = F.affine_grid(m, x.size(), align_corners=True).type(x.type())
        x = F.grid_sample(x, grid, align_corners=True, padding_mode="zeros", mode=mode)
        return x

    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    @staticmethod
    def rotate_cv_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @abc.abstractmethod
    def forward(self, x):
        pass

    def parameters(self, recurse: bool = True):
        return self.backbone.parameters(recurse)

    def train(self, mode: bool = True):
        super(RotationsModule, self).train(mode)
        self.backbone.train(mode)

    def eval(self):
        super(RotationsModule, self).eval()
        self.backbone.eval()


class AffordanceRotationsModule(RotationsModule):
    def forward(self, x, idxs=None, inference=True, softmax_at_the_end=True):
        # If backbone is of RotationsModule class, forward through it instead
        if hasattr(self.backbone, 'num_rotations'):
            return self.backbone(x, idxs=idxs, inference=inference)

        if not inference:
            self.train()
        else:
            self.eval()

        # If necessary append dummy batch dimension
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        assert len(x.shape) == 4

        if self.padding:
            # Add extra padding (to handle rotations inside network)
            diag_length = float(x.shape[2]) * np.sqrt(2)
            diag_length = np.ceil(diag_length / 32) * 32
            padding_width = int((diag_length - x.shape[2]) / 2)
            x = F.pad(x, (padding_width,) * 4)

        ys = []
        if idxs is None: idxs = range(self.num_rotations)
        for n in idxs:
            theta = n * self.max_angle / self.num_rotations
            if type(theta) == float: theta = torch.tensor(theta)

            # Rotate input for grasping angle inference
            rotated = x.clone()
            rotated = self.rotate_image(rotated, -theta)#.cuda()#to(self.backbone)

            if self.padding_noise > 0.:
                assert self.padding
                noise = torch.tensor(np.random.normal(0, self.padding_noise, rotated.shape), dtype=rotated.dtype, device=rotated.device)
                rotated[rotated == 0.] = (noise[rotated == 0.] - PickingDataset.mean[-1]) / PickingDataset.std[-1] if self.noise_normalization else noise[rotated==0.]
            else:
                rotated[rotated == 0.] = (rotated[rotated == 0.] - PickingDataset.mean[-1]) / PickingDataset.std[-1] if self.noise_normalization else rotated[rotated==0.]

            rotated = rotated.type(next(iter(self.backbone.parameters())).dtype).to(next(iter(self.backbone.parameters())).device)
            y = self.backbone(rotated)

            if isinstance(y, OrderedDict):
                y = y['out']

            # Rotate output back and unpad it
            y = self.rotate_image(y, theta)
            if self.padding:
                y = y[:, :, padding_width:-padding_width, padding_width:-padding_width]

            # ys.append(y.clone().detach().cpu() if inference else y)
            ys.append(y)

        shape = copy.copy(ys[0].shape) + (len(ys),)
        ys = torch.stack(ys, dim=len(shape) - 1).reshape(shape[0], -1)
        if softmax_at_the_end:
            ys = torch.nn.Softmax(dim=1)(ys)
        ys = ys.reshape(shape)

        return ys
