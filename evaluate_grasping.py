from light_grip.camera import UsbRealsenseCamera
from model.imitation_learning import ImitationLearning
from model.oracle import MonitoringAffordanceNetworkBasedOracle
from picking import Picking
from light_grip.robot import UR5eConfiguration
from light_grip.robot import UR5eTwoFinger


def main():
    model = ImitationLearning.load_from_checkpoint("model.ckpt").cuda()
    model.backbone.num_rotations = 16
    model.backbone.padding_noise = 0.0

    environment = UR5eTwoFinger(UR5eConfiguration())

    camera = UsbRealsenseCamera(heightmap_resolution=1 / 748, workspace=environment.workspace(), realtime=True)
    oracle = MonitoringAffordanceNetworkBasedOracle(camera, model.backbone, save_folder="vizualizations")
    pipeline = Picking(model.backbone, camera, environment, oracle, heightmap_transform=lambda x: x[3:, ...])

    for grasp in pipeline.loop():
        print(grasp.successful)


if __name__ == "__main__":
    main()
