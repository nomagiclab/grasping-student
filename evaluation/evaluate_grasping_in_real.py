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
    task = Task.init(project_name="krzywicki", task_name=f"final-robot-evaluation")

    teacher = ImitationLearning.load_from_checkpoint("../artifacts/teacher-0.2.ckpt", strict=False).cuda()
    teacher.backbone.num_rotations = 16
    teacher.backbone.padding_noise = 0.0

    student = StudentLearning.load_from_checkpoint("../artifacts/student-0.2.ckpt", strict=False).cuda()
    student.backbone.num_rotations = 16
    student.backbone.padding_noise = 0.0

    environment = UR5eTwoFinger(UR5eConfiguration(z_range=0.1, workspace_size=0.45 + 1/748, iters_tool_contact=1))

    camera = UsbRealsenseCamera(heightmap_resolution=1 / 748, workspace=environment.workspace(), realtime=True)
    oracle = MonitoringAffordanceNetworkBasedOracle(camera, None, save_folder=None)
    pipeline = Picking(None, camera, environment, oracle, heightmap_transform=lambda x: PickingDataset.normalize_depth(x[3:, ...]))

    epoch = 0
    current_model = teacher
    current_model_label = "teacher"
    for grasp in pipeline.loop():
        if grasp.epoch != epoch:
            epoch = grasp.epoch
            current_model = student if current_model_label == "teacher" else teacher
            current_model_label = "student" if current_model_label == "teacher" else "teacher"

        oracle.model = current_model.backbone
        pipeline.model = current_model.backbone

        with open(f'~/{current_model_label}/{datetime.now().strftime("%d-%m-%Y-%H_%M_%S")}', 'wb') as handle:
            task.get_logger().report_scalar(f"{current_model_label}-success-rate", "metrics", grasp.successful, grasp.iteration)
            pickle.dump(grasp, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
