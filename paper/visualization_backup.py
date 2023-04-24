import copy
import io
from datetime import datetime

import clearml
import cv2
import numpy as np
import torch
from clearml import Task
from matplotlib import pyplot as plt

from dataset.pointwise import PickingDataset, AugmentedPickingDataset
from light_grip.camera.realsense import UsbRealsenseCamera
from model.imitation_learning import ImitationLearning
from model.oracle import MonitoringAffordanceNetworkBasedOracle
from model.rotations import RotationsModule
from model.student_learning import StudentLearning
from light_grip.robot.abstract import UR5eConfiguration
from light_grip.robot.two_finger import UR5eTwoFinger
import PIL
import pickle


def get_matplotlib_numpy():
    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png", dpi=300)
    # f'input-{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.png'
    plt.savefig(f'input-{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.pdf', dpi=300)
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    ar = np.asarray(image)
    buffer_.close()
    plt.close()

    return ar


def predict(image, teacher, student, affordance):

    teacher.eval()
    student.eval()
    affordance.eval()
    with torch.no_grad():
        def obtain_affordances(model=teacher, image=image):
            affordances = model.backbone.forward(image, softmax_at_the_end=False)
            affordances = copy.deepcopy(affordances)
            return affordances

        num_base_images = 1
        fig, (axes) = plt.subplots(num_base_images, teacher.backbone.num_rotations)
        plt.tight_layout()

        if len(axes.shape) == 1:
            axes = axes.reshape(len(axes), 1)
        fig.set_size_inches(np.array((7, 2)) * 0.8)

        gs = [axes[x, 0].get_gridspec() for x in range(num_base_images, num_base_images)]
        metrics_axes = []
        for x in range(num_base_images, num_base_images):
            for ax in axes[x, :]:
                ax.remove()

            metrics_axes.append(fig.add_subplot(gs[x - num_base_images][x, :]))

        fig.tight_layout()
        heightmap_color = copy.deepcopy(image[:3])
        heightmap_depth = PickingDataset.normalize_depth(copy.deepcopy(image[3:]))

        for i, affordances in enumerate([obtain_affordances(model=m, image=heightmap_depth) for m in [affordance]]):
            affordances = torch.sigmoid(affordances)
            for angle_idx, ax in enumerate(axes):
                affordance = affordances[0, 0, :, :, angle_idx:(angle_idx + 1)].clone().detach().cpu().numpy()
                axes[angle_idx, i].set_title(f'{angle_idx / teacher.backbone.num_rotations}$\pi$')#, fontdict={"fontsize": 11})

                heightmap_color_rotated = RotationsModule.rotate_cv_image(np.transpose(heightmap_color.numpy(), (1, 2, 0)), -angle_idx / teacher.backbone.num_rotations * 180)
                # ax[0].imshow(heightmap_color_rotated)
                affordance_rotated = RotationsModule.rotate_cv_image(affordance, -angle_idx / teacher.backbone.num_rotations * 180)
                # affordance_rotated = (affordance_rotated - affordance_rotated.min()) / (affordance_rotated.max() - affordance_rotated.min())
                axes[angle_idx, i].set_axis_off()
                affordance_rotated[affordance_rotated < 0.8] = 0.
                heightmap_color_rotated[..., 1] += affordance_rotated
                # ax[i].imshow(affordance_rotated, alpha=0.4)
                axes[angle_idx, i].imshow(heightmap_color_rotated)
                # ax[i].imshow(affordance_rotated)#, alpha=0.6)
                plt.tight_layout()

    for ax, (title, fn, scale) in zip(metrics_axes):
        results = []
        for angle_idx in range(teacher.num_rotations):
            affordance = affordances[0, 0, :, :, angle_idx:(angle_idx + 1)].clone().detach().cpu().numpy()
            results.append(fn(affordance))

            ax.set_title(title)
            ax.set_yscale(scale)
            ax.bar(x=[f'{x / teacher.num_rotations}$\pi$' for x in range(teacher.num_rotations)], height=results)
        metrics_axes[-2].set_ylim(7.)
        plt.tight_layout()

    # fig.text(0.5, 0.04, 'common xlabel', ha='center', va='center')
    # fig.text(0.0085, 0.75, 'teacher', ha='center', va='center', rotation='vertical', fontdict={"fontsize": 11})
    # fig.text(0.0085, 0.25, 'student', ha='center', va='center', rotation='vertical', fontdict={"fontsize": 11})
    # fig.text(0.0085, 0.2, 'affordance', ha='center', va='center', rotation='vertical', fontdict={"fontsize": 11})

    affordances_viz = get_matplotlib_numpy()
    # print(1)
    # cv2.imshow("Network affordances", affordances_viz)
    # cv2.imwrite(f'input-{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.png', affordances_viz)
    # cv2.waitKey(1)


def main():
    teacher = ImitationLearning.load_from_checkpoint("../artifacts/teacher-0.2.ckpt", strict=False).cuda()
    teacher.backbone.num_rotations = 4
    teacher.affordance_model.backbone.num_rotations = 4
    teacher.backbone.padding_noise = 0.0

    student = StudentLearning.load_from_checkpoint("../artifacts/student-0.2.ckpt", strict=False).cuda()
    student.backbone.num_rotations = 4
    student.affordance_model.backbone.num_rotations = 4
    student.backbone.padding_noise = 0.0

    root_dir = clearml.Dataset.get(dataset_name="all-grasps", dataset_project="nomagic").get_local_copy()
    picking_dataset = PickingDataset(root_dir)

    for im in picking_dataset:
        predict(im["heightmap"], teacher, student, teacher.affordance_model)


if __name__ == "__main__":
    main()
