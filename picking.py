import math

import numpy as np
import torch

from dataset.schema import GraspingIndex, GraspingPoint, GraspAttempt


class Picking:
    def __init__(self, model, camera, environment, oracle, heightmap_transform=lambda x: x):
        self.model = model
        self.camera = camera
        self.environment = environment
        self.oracle = oracle
        self.heightmap_transform = heightmap_transform

    def loop(self):
        def grasping_idx_to_point(idx: GraspingIndex, z: float):
            workspace_limits = self.environment.workspace().limits()
            heightmap_shape = self.camera.heightmap_shape()
            return GraspingPoint(
                idx.angle_index * math.pi / self.model.num_rotations,
                workspace_limits[0, 0] + idx.row / heightmap_shape[0] * (workspace_limits[0, 1] - workspace_limits[0, 0]),
                workspace_limits[1, 0] + idx.col / heightmap_shape[0] * (workspace_limits[1, 1] - workspace_limits[1, 0]),
                z,
            )

        while True:
            # Get network input
            rgb, depth, segmentation = self.camera.take_photo()
            heightmap = self.camera.get_heightmap()
            raw_heightmap = heightmap.clone()

            if np.sum(depth > 0.) / np.sum(np.ones_like(depth)) < 0.005:
                self.environment.reset_bin_two_boxes()

            # Predict grasp
            affordance_output, grasping_index = self.oracle.predict(self.heightmap_transform(heightmap))
            grasping_point = grasping_idx_to_point(grasping_index, raw_heightmap[3, grasping_index.row, grasping_index.col])

            # Execute grasp
            successful = self.environment.grasp(grasping_point)
            yield GraspAttempt(
                heightmap, raw_heightmap,
                rgb, depth, segmentation,
                grasping_index, grasping_point,
                successful,
                self.camera.intrinsics(),
                self.camera.pose(),
                None,
                None,
                self.model.num_rotations,
            )
