import cv2
import numpy as np
import torch

from dataset.schema import HeightMapImage, GraspingIndex


class TwoFingerManualOracle:
    def __init__(self, gripper_thickness, gripper_width, gripper_spacing, num_rotations, workspace):
        self.gripper_thickness = gripper_thickness
        self.gripper_width = gripper_width
        self.gripper_spacing = gripper_spacing
        self.num_rotations = num_rotations
        self.workspace = workspace

        self.gripper_thickness_pix = None
        self.gripper_width_pix = None
        self.gripper_spacing_pix = None

        self.angle = 0.
        self.row = None
        self.col = None
        self.clicked = False

        self.image_size = None
        self.overlay = None

        def mouseclick_callback(event, c, r, flags, params):
            nonlocal self
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked = True

            if event == cv2.EVENT_MOUSEMOVE:
                self.col, self.row = c, r
                self.overlay = self.get_overlay()

        def on_trackbar(val):
            self.angle = val
            if self.col:
                self.overlay = self.get_overlay()

        cv2.namedWindow('Choose grasp-point')
        cv2.setMouseCallback('Choose grasp-point', mouseclick_callback)
        cv2.createTrackbar("Angle", 'Choose grasp-point', 0, num_rotations, on_trackbar)

    def reset(self):
        self.angle = 0.
        self.row = None
        self.col = None
        self.clicked = False
        self.overlay = None

    def get_overlay(self, val=None):
        if val is None: val = self.angle
        padding = 500
        overlay = np.ones((self.image_size + padding * 2, self.image_size + padding * 2), dtype=np.uint8) * 255
        col = self.col + padding
        row = self.row + padding

        left = col - self.gripper_spacing_pix // 2
        right = col + self.gripper_spacing_pix // 2

        def mark(pos):
            overlay[
                row-self.gripper_width_pix//2:row+self.gripper_width_pix//2,
                pos-self.gripper_thickness_pix//2:pos+self.gripper_thickness_pix//2,
            ] = 0

        mark(left)
        mark(right)

        angle = self.angle / self.num_rotations * 180.
        rot_mat = cv2.getRotationMatrix2D((col, row), angle, 1.0)
        overlay = cv2.warpAffine(overlay, rot_mat, overlay.shape, flags=cv2.INTER_LINEAR)

        return overlay[padding:-padding, padding:-padding]

    def predict(self, image: HeightMapImage):
        self.reset()

        image = torch.permute(image[:3, ...], (1, 2, 0))
        image = (image.numpy() * 255).astype(np.uint8)

        if self.gripper_thickness_pix is None:
            self.image_size = image.shape[0]
            self.gripper_thickness_pix = int(self.gripper_thickness / self.workspace.width() * self.image_size)
            self.gripper_width_pix = int(self.gripper_width / self.workspace.width() * self.image_size)
            self.gripper_spacing_pix = int(self.gripper_spacing / self.workspace.width() * self.image_size)

        while not self.clicked:
            if self.overlay is None:
                cv2.imshow("Choose grasp-point", image)
            else:
                overlay_rgb = cv2.cvtColor(self.overlay, cv2.COLOR_GRAY2RGB)
                cv2.imshow("Choose grasp-point", cv2.addWeighted(image, 0.8, overlay_rgb, 0.2, 1.))

            key = cv2.waitKey(1)
            if key == ord('q'):
                pos = (cv2.getTrackbarPos("Angle", 'Choose grasp-point') + 1) % self.num_rotations
                cv2.setTrackbarPos("Angle", 'Choose grasp-point', pos)
            elif key == ord('e'):
                pos = (cv2.getTrackbarPos("Angle", 'Choose grasp-point') - 1 + self.num_rotations) % self.num_rotations
                cv2.setTrackbarPos("Angle", 'Choose grasp-point', pos)

        return None, GraspingIndex(self.angle, self.row, self.col)
