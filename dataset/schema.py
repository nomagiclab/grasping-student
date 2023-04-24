from typing import NamedTuple, Union, List, Optional, Tuple

import numpy as np
import torch

""" First 3 coordinates on axis=0 are rgb, last coordinates on this axis is/are depth. """
HeightMapImage = torch.Tensor


class GraspingIndex(NamedTuple):
    """ For batching purposes we are allowing the fields to be lists. """
    angle_index: Union[int, List[int]]
    row: Union[int, List[int]]
    col: Union[int, List[int]]


class GraspingPoint(NamedTuple):
    """ For batching purposes we are allowing the fields to be lists. """
    angle: Union[float, List[float]]
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    z: Union[float, List[float]]


class GraspAttempt(NamedTuple):
    """ For batching purposes we are allowing the fields to be lists. """
    heightmap: Union[HeightMapImage, List[HeightMapImage]]
    raw_heightmap: Union[Optional[HeightMapImage], List[HeightMapImage]]

    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]

    segmentation: Union[Optional[torch.Tensor], List[Optional[torch.Tensor]]]
    grasping_index: Union[Optional[GraspingIndex], List[GraspingIndex]]
    grasping_point: Union[Optional[GraspingPoint], List[GraspingPoint]]
    successful: Union[Optional[bool], List[bool]]

    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[np.ndarray] = None

    rgb_normalization: Optional[Tuple[List[float], List[float]]] = None
    depth_normalization: Optional[Tuple[float, float]] = None

    num_rotations: Optional[int] = None

    affordances: Optional[torch.Tensor] = None
    iteration: Optional[int] = None
    epoch: Optional[int] = None
