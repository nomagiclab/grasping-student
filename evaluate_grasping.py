from camera.realsense import UsbRealsenseCamera
from dataset.pointwise import PickingDataset
from model.imitation_learning import ImitationLearning
from model.oracle import MonitoringAffordanceNetworkBasedOracle, RandomOracle
from picking import Picking
from robot.abstract import UR5eConfiguration
from robot.two_finger import UR5eTwoFinger


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
