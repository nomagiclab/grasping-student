import datetime
import os
import pickle

from light_grip.camera import UsbRealsenseCamera
from dataset.labelling import TwoFingerManualOracle
from model.imitation_learning import ImitationLearning
from picking import Picking
from light_grip.robot import UR5eConfiguration, AlwaysSuccessfulUR5eMock


def main():
    model = ImitationLearning.load_from_checkpoint("model.ckpt").cuda()
    model.backbone.num_rotations = 64
    model.backbone.padding_noise = 0.0

    # environment = UR5eTwoFinger(UR5eConfiguration(
    #     skip_grasps_with_low_depth=False, z_range=0.25,
    #     iters_tool_contact=50, max_finger_diff_object_detection=100,
    #     reset_bin=False,
    #     always_non_empty=True,
    #     place_bin_positions=[
    #         [0.4665111185116722, -0.4613840230729267, 0.4371465097775752, -0.6564793184935471, 2.653525768589614,
    #          -0.5350808193310717],
    #         [0.10651404146904263, -0.9034469707095425, 0.29957833367474154, -0.6229379331060064, -2.874448502734379,
    #          1.0675311375135699],
    #     ]
    # ))
    environment = AlwaysSuccessfulUR5eMock(UR5eConfiguration(z_range=0.25))

    camera = UsbRealsenseCamera(heightmap_resolution=1 / 748, workspace=environment.workspace(), realtime=True)
    oracle = TwoFingerManualOracle(0.005, 0.025, 0.085, model.backbone.num_rotations, environment.workspace())
    pipeline = Picking(model.backbone, camera, environment, oracle)

    dir_name = f"dataset/manual-data-{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.system(f"mkdir -p {dir_name}")

    for n, grasp in enumerate(pipeline.loop()):
        def dictify(x):
            assert hasattr(x, '_asdict')
            return {k: dictify(v) if hasattr(v, '_asdict') else v for k, v in x._asdict().items()}

        with open(os.path.join(dir_name, f'{n}.pkl'), 'wb') as f:
            pickle.dump([dictify(grasp)], f)


if __name__ == "__main__":
    main()
