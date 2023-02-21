from datetime import datetime

from clearml import Task

from dataset.pointwise import PickingDataset
from light_grip.camera.realsense import UsbRealsenseCamera
from model.imitation_learning import ImitationLearning
from model.oracle import MonitoringAffordanceNetworkBasedOracle
from model.student_learning import StudentLearning
from picking import Picking
from light_grip.robot.abstract import UR5eConfiguration
from light_grip.robot.two_finger import UR5eTwoFinger
import pickle


def main():
    task = Task.init(project_name="krzywicki", task_name=f"final-robot-evaluation-teacher-no-triangles-dense")

    model = StudentLearning.load_from_checkpoint("../artifacts/teacher-1.0.ckpt", strict=False).cuda()  # 90%
    model.backbone.num_rotations = 16
    model.backbone.padding_noise = 0.0

    environment = UR5eTwoFinger(UR5eConfiguration(z_range=0.1, workspace_size=0.45 + 1/748, iters_tool_contact=1))

    camera = UsbRealsenseCamera(heightmap_resolution=1 / 748, workspace=environment.workspace(), realtime=True)
    oracle = MonitoringAffordanceNetworkBasedOracle(camera, model.backbone, save_folder=None)
    pipeline = Picking(model.backbone, camera, environment, oracle, heightmap_transform=lambda x: PickingDataset.normalize_depth(x[3:, ...]))

    for grasp in pipeline.loop():
        with open(f'workbench/{datetime.now().strftime("%d-%m-%Y-%H_%M_%S")}', 'wb') as handle:
            task.get_logger().report_scalar("success-rate", "metrics", grasp.successful, grasp.iteration)
            pickle.dump(grasp, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
