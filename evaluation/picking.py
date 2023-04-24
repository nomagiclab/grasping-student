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

        epoch = 0
        iteration = 0
        result = []
        while True:
            # Get network input
            # rgb, depth, segmentation = self.camera.take_photo()
            heightmap = self.camera.get_heightmap()
            margins = self.camera.calculate_margins()
            raw_heightmap = heightmap.clone()
            heightmap[3, heightmap[3] < 0.002] = 0.
            allowed_heightmap = heightmap[:, margins[0]:-margins[0], margins[1]:-margins[1]]
            if torch.sum((allowed_heightmap[3]) > 0.0) / torch.sum(torch.ones_like(allowed_heightmap[3])) < 0.02:
                self.environment.reset_bin_two_boxes()
                epoch += 1
                yield GraspAttempt(
                    heightmap=heightmap,
                    raw_heightmap=raw_heightmap,
                    rgb=None,
                    depth=None,
                    segmentation=None,
                    grasping_index=None,
                    grasping_point=None,
                    successful=None,
                    camera_intrinsics=None,
                    camera_extrinsics=None,
                    rgb_normalization=None,
                    depth_normalization=None,
                    num_rotations=None,
                    affordances=None,
                    epoch=epoch,
                    iteration=iteration,
                )
                continue

            # Predict grasp
            affordance_output, grasping_index = self.oracle.predict(self.heightmap_transform(heightmap))
            grasping_point = grasping_idx_to_point(grasping_index, torch.mean(raw_heightmap[3, grasping_index.row-1:grasping_index.row+1, grasping_index.col-1:grasping_index.col+1]).item())

            # Execute grasp
            successful = self.environment.grasp(grasping_point)
            iteration += 1
            result.append(successful)

            if len(result) >= 5 and np.sum(result[-5:]) == 0:
                self.environment.reset_bin_two_boxes()
                yield GraspAttempt(
                    heightmap=heightmap,
                    raw_heightmap=raw_heightmap,
                    rgb=None,
                    depth=None,
                    segmentation=None,
                    grasping_index=grasping_index,
                    grasping_point=grasping_point,
                    successful=successful,
                    camera_intrinsics=None,
                    camera_extrinsics=None,
                    rgb_normalization=None,
                    depth_normalization=None,
                    num_rotations=self.model.num_rotations,
                    affordances=None,
                    epoch=epoch,
                    iteration=iteration,
                )
                epoch += 1
                yield GraspAttempt(
                    heightmap=heightmap,
                    raw_heightmap=raw_heightmap,
                    rgb=None,
                    depth=None,
                    segmentation=None,
                    grasping_index=None,
                    grasping_point=None,
                    successful=None,
                    camera_intrinsics=None,
                    camera_extrinsics=None,
                    rgb_normalization=None,
                    depth_normalization=None,
                    num_rotations=None,
                    affordances=None,
                    epoch=epoch,
                    iteration=iteration,
                )
                result = []
                continue

            yield GraspAttempt(
                heightmap=heightmap,
                raw_heightmap=raw_heightmap,
                rgb=None,
                depth=None,
                segmentation=None,
                grasping_index=grasping_index,
                grasping_point=grasping_point,
                successful=successful,
                camera_intrinsics=None,
                camera_extrinsics=None,
                rgb_normalization=None,
                depth_normalization=None,
                num_rotations=self.model.num_rotations,
                affordances=None,
                epoch=epoch,
                iteration=iteration,
            )
