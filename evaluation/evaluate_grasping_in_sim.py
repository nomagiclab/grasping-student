import pickle
from datetime import datetime

from clearml import Task

from camera.pybullet_camera import PyBulletCameraService
from dataset.pointwise import PickingDataset
from light_grip.camera.realsense import UsbRealsenseCamera
from model.imitation_learning import ImitationLearning
from model.oracle import MonitoringAffordanceNetworkBasedOracle
from picking import Picking
from bullet.base_simulator import PyBulletSimulator, PyBulletSimulatorConfiguration


def main():
    task = Task.init(project_name="krzywicki", task_name=f"final-sim-evaluation-teacher")

    model = ImitationLearning.load_from_checkpoint("../artifacts/teacher-1.0.ckpt").cuda()
    model.backbone.num_rotations = 16
    model.backbone.padding_noise = 0.0

    environment = PyBulletSimulator(PyBulletSimulatorConfiguration(gui=True, spread=0.4, object_count=50))
    environment.reset()

    camera = PyBulletCameraService(
        eye=[0.5, 0.5, 5],
        rectangle=[
            [1., 0., 0],
            [1., 1., 0],
            [0., 1., 0],
            [0., 0., 0],
        ],
        near_scale=0.4, far_scale=4,
    )
    oracle = MonitoringAffordanceNetworkBasedOracle(camera, model.backbone, verbose=False)
    pipeline = Picking(model.backbone, camera, environment, oracle, heightmap_transform=lambda x: PickingDataset.normalize_depth(x[3:, ...]))

    for grasp in pipeline.loop():
        # with open(f'sim/{datetime.now().strftime("%d-%m-%Y-%H_%M_%S")}', 'wb') as handle:
        task.get_logger().report_scalar("success-rate", "metrics", grasp.successful, grasp.iteration)
        # pickle.dump(grasp, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
