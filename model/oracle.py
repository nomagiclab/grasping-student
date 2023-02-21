import copy
import datetime
import io
import random

import PIL
import cv2
import numpy as np

import torch

import matplotlib.pyplot as plt

from dataset.schema import HeightMapImage, GraspingIndex
from model.rotations import RotationsModule


class AffordanceNetworkBasedOracle:
    def __init__(self, model):
        self.model = model

    # Effectively: theta, x, y = argmax over theta, x, y of  affordances
    @staticmethod
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    @staticmethod
    def select(heightmap, affordances):
        with torch.no_grad():
            affordances = copy.deepcopy(affordances)
            depth = heightmap[3, ...] if heightmap.shape[0] > 1 else heightmap[0]
            affordances[0, 0, depth <= depth.nanmean()] = 0
            affordances[0, 0, depth <= 0.] = 0
            assert torch.sum(affordances) > 0.

            flat = affordances.flatten()
            threshold = torch.sort(flat)[0][int(0.99 * len(flat))]
            flat[flat < threshold] = 0.

            if torch.sum(affordances) > 0.:
                affordances /= torch.sum(affordances)

            idx = torch.multinomial(affordances.flatten(), 1).item()
            _, _, xpixel, ypixel, thetaprim = AffordanceNetworkBasedOracle.unravel_index(idx, affordances.shape)
            return GraspingIndex(thetaprim, xpixel, ypixel)

    @staticmethod
    def select_argmax(heightmap, affordances):
        idx = torch.argmax(affordances).item()
        _, _, row, col, angle_index = AffordanceNetworkBasedOracle.unravel_index(idx, affordances.shape)
        return GraspingIndex(angle_index, row, col)

    def predict(self, image: HeightMapImage):
        self.model.eval()
        with torch.no_grad():
            affordances = self.model(image)

        grasping_point = self.select(image, affordances)
        return affordances, grasping_point


class RandomOracle:
    def __init__(self, num_rotations):
        self.num_rotations = num_rotations

    def predict(self, image: HeightMapImage):
        _, height, width = image.shape
        return None, GraspingIndex(
            random.randint(0, self.num_rotations - 1),
            random.randint(0, height - 1),
            random.randint(0, width - 1),
        )


class MonitoringAffordanceNetworkBasedOracle:
    def __init__(self, camera, model, save_folder=None, verbose=True):
        self.camera = camera
        self.model = model
        self.save_folder = save_folder
        self.verbose = verbose
        if self.verbose:
            # Hack because of opencv bug – first imshow is blank
            cv2.imshow("Network affordances", np.zeros((1, 1)))
            cv2.waitKey(1)

            cv2.imshow("Neural network input", np.zeros((1, 1)))
            cv2.waitKey(1)

    @staticmethod
    def get_matplotlib_numpy():
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)
        ar = np.asarray(image)
        buffer_.close()
        plt.close()

        return ar

    def predict(self, image: HeightMapImage):
        rgb, depth = self.camera.raw_photo(colorized_depth=False)
        full_heightmap = self.camera.get_heightmap()
        if self.verbose:
            def torch_to_cv_image(heightmap):
                return torch.permute(heightmap, (1, 2, 0)).clone().detach().cpu().numpy()

            heightmap_color = torch_to_cv_image(full_heightmap[:3, ...])
            heightmap_depth = torch_to_cv_image(full_heightmap[3:, ...])

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            ax1.set_title("rgb")
            ax1.imshow(rgb)

            ax2.set_title("depth")
            ax2.imshow(depth)

            ax3.set_title("heightmap-rgb")
            ax3.imshow(heightmap_color)

            ax4.set_title("heightmap-depth")
            ax4.imshow(heightmap_depth)
            plt.tight_layout()

            input_viz = self.get_matplotlib_numpy()
            cv2.imshow("Neural network input", input_viz)
            if self.save_folder:
                cv2.imwrite(f'{self.save_folder}/input-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.png', input_viz)
            cv2.waitKey(1)

        self.model.eval()
        with torch.no_grad():
            affordances = self.model.forward(image)
            # affordances = self.model.forward(image, idxs=[x * torch.ones((h.shape[0])).to(self.device_) for x in list(range())])
            # affordances = self.model(torch.unsqueeze(image, dim=0).repeat((self.model.num_rotations, 1, 1, 1)), idxs=[torch.tensor(list(range(self.model.num_rotations)), device=image.device)])
            # affordances = torch.permute(affordances, dims=(1, 4, 2, 3, 0))


            # masks = np.expand_dims(self.camera.calculate_mask(), axis=(0))
            # masks = torch.from_numpy(masks).to(image.device)
            #
            # inner_affordances = self.model(torch.unsqueeze(image * masks, dim=0).repeat((self.model.num_rotations, 1, 1, 1)), idxs=[torch.tensor(list(range(self.model.num_rotations)), device=image.device)], softmax_at_the_end=True)
            # inner_affordances = torch.permute(inner_affordances, dims=(1, 4, 2, 3, 0))

            affordances = copy.deepcopy(affordances)
            d = full_heightmap[3, ...] if full_heightmap.shape[0] > 1 else full_heightmap[0]
            affordances[0, 0, d <= d.nanmean()] = 0
            affordances[0, 0, d <= 0.] = 0
            assert torch.sum(affordances) > 0.
            if torch.sum(affordances) > 0.:
                affordances /= torch.sum(affordances)

        def total_affordance(affordance):
            return np.sum(affordance)

        def max_affordance(affordance):
            return np.max(affordance)

        def entropy(affordance):
            if np.sum(affordance) == 0.:
                return 0.
            affordance = affordance / np.sum(affordance)
            return np.sum(-np.ma.log(affordance).filled(0.) * affordance)

        def percentile(affordance, percentile=0.99):
            affordance = affordance.flatten()
            return np.sort(affordance)[int(len(affordance) * percentile)]

        metrics = [
            ('Total affordance', total_affordance, 'log'),
            ('Max affordance', max_affordance, 'linear'),
            # ('Entropy', entropy, 'linear'),
            # ('0.99-percentile', percentile, 'linear'),
            # ('median', lambda a: percentile(a, percentile=0.5), 'linear'),
        ]

        num_base_images = 4
        if self.verbose:
            fig, (axes) = plt.subplots(len(metrics) + num_base_images, self.model.num_rotations)
            if len(axes.shape) == 1:
                axes = axes.reshape(len(axes), 1)
            fig.set_size_inches(np.array((2 * self.model.num_rotations + 2, 2 * (len(metrics) + num_base_images) + 2)) * 0.8)

            gs = [axes[x, 0].get_gridspec() for x in range(num_base_images, len(metrics) + num_base_images)]
            metrics_axes = []
            for x in range(num_base_images, len(metrics) + num_base_images):
                for ax in axes[x, :]:
                    ax.remove()

                metrics_axes.append(fig.add_subplot(gs[x - num_base_images][x, :]))

            fig.tight_layout()

            if affordances is not None:
                for angle_idx, ax in enumerate(axes.T):
                    affordance = affordances[0, 0, :, :, angle_idx:(angle_idx + 1)].clone().detach().cpu().numpy()
                    ax[0].set_title(f'angle={angle_idx / self.model.num_rotations}$\pi$')

                    heightmap_color_rotated = RotationsModule.rotate_cv_image(heightmap_color, -angle_idx / self.model.num_rotations * 180)
                    heightmap_depth_rotated = RotationsModule.rotate_cv_image(heightmap_depth, -angle_idx / self.model.num_rotations * 180)
                    affordance_rotated = RotationsModule.rotate_cv_image(affordance, -angle_idx / self.model.num_rotations * 180)

                    ax[0].imshow(heightmap_color_rotated)
                    ax[1].imshow(heightmap_depth_rotated)
                    ax[2].imshow(affordance_rotated)
                    ax[3].imshow(affordance_rotated, alpha=0.8)
                    ax[3].imshow(heightmap_color_rotated, alpha=0.2)

            for ax, (title, fn, scale) in zip(metrics_axes, metrics):
                results = []
                for angle_idx in range(self.model.num_rotations):
                    affordance = affordances[0, 0, :, :, angle_idx:(angle_idx + 1)].clone().detach().cpu().numpy()
                    results.append(fn(affordance))

                ax.set_title(title)
                ax.set_yscale(scale)
                ax.bar(x=[f'{x / self.model.num_rotations}$\pi$' for x in range(self.model.num_rotations)], height=results)
            metrics_axes[-2].set_ylim(7.)
            plt.tight_layout()

            affordances_viz = self.get_matplotlib_numpy()
            cv2.imshow("Network affordances", affordances_viz)
            if self.save_folder:
                cv2.imwrite(f'{self.save_folder}/affordances-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.png', affordances_viz)
            cv2.waitKey(1)

        masks = np.expand_dims(self.camera.calculate_mask(), axis=(0, 1, 4))
        masks = torch.from_numpy(masks).to(affordances.device)
        grasping_point = AffordanceNetworkBasedOracle.select_argmax(image, affordances * masks)
        return affordances, grasping_point
