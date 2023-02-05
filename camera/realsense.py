from typing import Tuple

import numpy as np
import pyrealsense2 as rs
import torch
import torchvision

from dataset.schema import HeightMapImage


class UsbRealsenseCamera:
    def __init__(self, heightmap_resolution, workspace, realtime=False):
        super().__init__()
        self.realtime = realtime
        self.pipe = None
        self.cfg = None
        self.profile = None
        self.heightmap_resolution = heightmap_resolution
        self.workspace = workspace

        self.init_pipeline()

    def init_pipeline(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, *self.shape())
        self.cfg.enable_stream(rs.stream.depth, *self.shape())
        self.profile = self.pipe.start(self.cfg)

        # Skip 10 first frames to give the Auto-Exposure time to adjust
        for x in range(10):
            self.pipe.wait_for_frames()

    def raw_realsense_photo(self):
        if not self.realtime:
            self.init_pipeline()

        # Store next frameset for later processing:
        frameset = self.pipe.wait_for_frames()

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        color_frame = frameset.get_color_frame()

        aligned_depth_frame = frameset.get_depth_frame()
        return color_frame, aligned_depth_frame

    def raw_photo(self, colorized_depth=False):
        color_frame, aligned_depth_frame = self.raw_realsense_photo()
        color = np.asanyarray(color_frame.get_data())
        if colorized_depth:
            colorizer = rs.colorizer()
            aligned_depth_frame = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        else:
            aligned_depth_frame = np.asanyarray(aligned_depth_frame.get_data())

        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        return color, aligned_depth_frame * depth_scale

    def intrinsics(self):
        return np.array([
            [909.4387817382812, 0.0,               632.1474609375],
            [0.0,               907.5307006835938, 347.2157897949219],
            [0.0,               0.0,               1.0],
        ])

    def pose(self):
        cam_pose = np.loadtxt('robot/camera_pose.txt', delimiter=' ')
        return cam_pose[0:3, 0:3], cam_pose[0:3, 3]

    def shape(self) -> Tuple[int, int]:
        return 1280, 720

    def take_photo(self):
        color, depth = self.raw_photo(colorized_depth=False)
        return color, depth, None

    def get_pointcloud(self):
        rgb, depth, _ = self.take_photo()
        height, width = depth.shape

        intrinsics = self.intrinsics()

        # Mapping from 2D image coordinates to coordinates in 3D from camera origin
        xs, ys = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
        xs = np.multiply(xs - intrinsics[0][2], depth / intrinsics[0][0]).flatten()
        ys = np.multiply(ys - intrinsics[1][2], depth / intrinsics[1][1]).flatten()
        zs = depth.copy().flatten()

        rs, gs, bs = rgb[:, :, 0].flatten(), rgb[:, :, 1].flatten(), rgb[:, :, 2].flatten()
        xyz, rgb = np.stack([xs, ys, zs], axis=1), np.stack([rs, gs, bs], axis=1)

        # Transform 3D point cloud from camera coordinates to robot coordinates
        r, t = self.pose()
        xyz = np.transpose(np.dot(r, np.transpose(xyz)) + np.tile(t.reshape(3, 1), (1, xyz.shape[0])))

        return xyz, rgb

    def heightmap_shape(self):
        return np.ceil((
            self.workspace.height() / self.heightmap_resolution,
            self.workspace.width() / self.heightmap_resolution
        )).astype(int)

    def get_heightmap(self):
        xyz, rgb = self.get_pointcloud()

        sz = self.heightmap_shape()

        # Sort surface points by z value
        zs = np.argsort(xyz[:, 2])
        xyz = xyz[zs]
        rgb = rgb[zs]

        # Filter out surface points outside heightmap boundaries
        valid = np.logical_and(np.logical_and(np.logical_and(np.logical_and(
            xyz[:, 0] >= self.workspace.limits()[0][0], xyz[:, 0] < self.workspace.limits()[0][1]),
            xyz[:, 1] >= self.workspace.limits()[1][0]), xyz[:, 1] < self.workspace.limits()[1][1]),
            xyz[:, 2] < self.workspace.limits()[2][1]
        )
        xyz = xyz[valid]
        rgb = rgb[valid]

        # Create orthographic top-down-view RGB-D heightmaps
        rows = np.floor((xyz[:, 0] - self.workspace.limits()[0][0]) / self.heightmap_resolution).astype(int)
        cols = np.floor((xyz[:, 1] - self.workspace.limits()[1][0]) / self.heightmap_resolution).astype(int)

        rs = np.zeros((sz[0], sz[1], 1), dtype=np.uint8)
        gs = np.zeros((sz[0], sz[1], 1), dtype=np.uint8)
        bs = np.zeros((sz[0], sz[1], 1), dtype=np.uint8)

        rs[rows, cols] = rgb[:, [0]]
        gs[rows, cols] = rgb[:, [1]]
        bs[rows, cols] = rgb[:, [2]]

        color_heightmap = np.concatenate((rs, gs, bs), axis=2)

        depth_heightmap = np.zeros(sz)
        depth_heightmap[rows, cols] = xyz[:, 2]

        z_bottom = self.workspace.limits()[2][0]
        depth_heightmap[depth_heightmap != 0] = depth_heightmap[depth_heightmap != 0] - z_bottom
        depth_heightmap[depth_heightmap <= 0] = 0.

        def cv_to_heightmap(color: np.ndarray, depth: np.ndarray) -> HeightMapImage:
            return torch.cat([
                torchvision.transforms.ToTensor()(color),
                torch.unsqueeze(torch.from_numpy(depth), dim=0)
            ])

        return cv_to_heightmap(color_heightmap, depth_heightmap)

    def __del__(self):
        self.pipe.stop()
