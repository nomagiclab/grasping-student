import copy
from enum import Enum
from typing import List, NamedTuple

import numpy as np
import logging
import rtde_control
import rtde_receive

from dataset.schema import GraspingPoint
from robot.guarded_rtde import GuardedRtde


class GraspingWorkspace:
    def __init__(self, limits: List[List[float]]):
        self.limits_ = np.array(limits)

    def width(self):
        return self.limits()[0, 1] - self.limits()[0, 0]

    def height(self):
        return self.limits()[1, 1] - self.limits()[1, 0]

    def depth(self):
        return self.limits()[2, 1] - self.limits()[2, 0]

    def reference_point(self):
        return self.limits()[:3, 0]

    def limits(self) -> np.ndarray:
        return self.limits_


class MoveType(Enum):
    MOVEL = 1
    MOVEJ = 2


class UR5eConfiguration(NamedTuple):
    robot_ip: str = "192.168.1.20"
    home: List[float] = [-0.013460461293355763, -1.2448845368674775, -1.8535711765289307, -1.6164161167540492, 1.5796574354171753, 0.05403100699186325]
    speed: float = 1.
    acceleration: float = 2.
    activate_gripper: bool = True
    gripper_force: int = 100
    gripper_speed: int = 100
    iters_tool_contact: int = 1
    max_finger_diff_object_detection: int = 25
    move_type: MoveType = MoveType.MOVEJ
    payload: float = 0.5
    payload_offset: List[float] = [0., 0., 0.1628]
    tcp_offset: List[float] = [0., 0., 0.1628]
    reset_bin_position: List[List[float]] = [0.428, -0.342, 0.078, 2.1420118296214126, -2.296328155967399, -0.0016354965825154653]
    place_bin_positions: List[List[float]] = [[0.428, -0.25, 0.238, 2.1420118296214126, -2.296328155967399, -0.0016354965825154653]]
    workspace_size: float = 0.4
    reference_point: List[float] = [0.525, 0.26, -0.0275]
    z_range: float = 0.05
    bin_width: float = 0.3
    bin_height: float = 0.3
    initialize: bool = True
    skip_grasps_with_low_depth: bool = True
    return_at_center: bool = False
    opened_gripper_position: int = 30
    reset_bin: bool = True
    skip_reset_bin_at_first_grasp: bool = True
    skip_calibration: bool = False
    always_non_empty: bool = False


class UR5e:
    def grasp(self, grasping_point: GraspingPoint) -> bool:
        pass

    def __init__(self, config: UR5eConfiguration):
        self.config = config
        if config.initialize:
            self.unsafe_rtde_ctl = rtde_control.RTDEControlInterface(self.config.robot_ip)
            self.rtde_ctl = GuardedRtde(self.unsafe_rtde_ctl)

            self.rtde_ctl.setPayload(self.config.payload, self.config.payload_offset)

            self.rtde_info = rtde_receive.RTDEReceiveInterface(self.config.robot_ip)
            assert self.rtde_info

            self.move_home()

    def get_home(self):
        return copy.deepcopy(self.config.home)

    def move_home(self):
        home_pose = copy.deepcopy(self.config.home)
        home_pose[5:] = self.rtde_info.getActualQ()[5:]
        self.move(home_pose, cart=self.config.move_type == MoveType.MOVEL)
        self.rtde_ctl.zeroFtSensor()

    def place_to_bin(self):
        if self.config.return_at_center:
            place_pos = self.rtde_info.getActualTCPPose()
            place_pos[:3] = [
                self.config.reference_point[0],
                self.config.reference_point[1],
                self.config.reference_point[2] + self.config.z_range
            ]
            self.move(place_pos)

        else:
            upper_pose = self.rtde_info.getActualTCPPose()
            upper_pose[2] += 0.1

            q = self.rtde_info.getActualQ()

            for position in [upper_pose] + self.config.place_bin_positions:
                position = self.rtde_ctl.getInverseKinematics(position)
                position[5:] = q[5:]
                self.move(position, cart=False)

        self.release()

    def move(self, coord, speed=None, acceleration=None, cart=True, asynchronous=False):
        if speed is None: speed = self.config.speed
        if acceleration is None: acceleration = self.config.acceleration
        resolver = {
            (MoveType.MOVEJ, True): self.rtde_ctl.moveJ_IK,
            (MoveType.MOVEJ, False): self.rtde_ctl.moveJ,
            (MoveType.MOVEL, True): self.rtde_ctl.moveL,
            (MoveType.MOVEL, False): self.rtde_ctl.moveL_FK,
        }
        resolver[(self.config.move_type, cart)](coord, asynchronous=asynchronous, speed=speed, acceleration=acceleration)

    # WARNING: This way of detecting tool contact doesn't always work,
    # use move_guarded_urscript()
    def move_guarded(self, target, tolerance=0.001, speed=None, move_up=True):
        self.move(target, asynchronous=True)

        while True:
            actual = self.rtde_info.getActualTCPPose()
            if np.linalg.norm(np.array(actual) - np.array(target)) < tolerance:
                return False

            if self.unsafe_rtde_ctl.toolContact([0., 0., 0., 0., 0., 0.]) != 0:
                if move_up:
                    actual[2] += 0.05
                    self.move(actual)

                logging.info("Stopping, force on tcp detected.")
                return True

    def move_guarded_urscript(self, target, tolerance=0.001, speed=None, acceleration=None, move_up=True):
        if not speed: speed = self.config.speed
        if not acceleration: acceleration = self.config.acceleration
        formatted_target = [float("{0:0.3f}".format(fl)) for fl in target]
        script = f""" 
thread movel_job():
    movel(p{formatted_target}, {speed}, {acceleration})
end

job = run movel_job()
write_output_integer_register(19, 0)

while True:
    actual = get_actual_tcp_pose()

    if pose_dist(actual, p{formatted_target}) < {tolerance}:
        kill job
        stopl(10.0)
        break
    end

    if tool_contact(p[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) >= {self.config.iters_tool_contact}:
        write_output_integer_register(19, 1)
        actual[2] = actual[2] + 0.05
        kill job
        stopl(10.0)
        if {move_up}:
            movel(actual)
        end
        break
    end
    sync()
end
"""
        self.rtde_ctl.sendCustomScriptFunction("MOVEL_GUARDED", script)
        return bool(self.rtde_info.getOutputIntRegister(19))

    def reset(self, seed=None) -> None:
        self.unsafe_rtde_ctl.disconnect()
        self.rtde_info.disconnect()
        del self.unsafe_rtde_ctl
        del self.rtde_info
        self.unsafe_rtde_ctl = rtde_control.RTDEControlInterface(self.config.robot_ip)
        self.rtde_ctl = GuardedRtde(self.unsafe_rtde_ctl)
        self.rtde_ctl.setPayload(self.config.payload, self.config.payload_offset)
        self.rtde_info = rtde_receive.RTDEReceiveInterface(self.config.robot_ip)

        self.move_home()
        if not self.config.skip_reset_bin_at_first_grasp:
            self.reset_bin_two_boxes()

        self.config.skip_reset_bin_at_first_grasp = False

    def reset_bin(self):
        pass

    def release(self):
        pass

    def workspace(self) -> GraspingWorkspace:
        return GraspingWorkspace([
            [self.config.reference_point[0] - self.config.workspace_size / 2, self.config.reference_point[0] + self.config.workspace_size / 2],
            [self.config.reference_point[1] - self.config.workspace_size / 2, self.config.reference_point[1] + self.config.workspace_size / 2],
            [self.config.reference_point[2], self.config.reference_point[2] + self.config.z_range]
        ])


class AlwaysSuccessfulUR5eMock:
    def __init__(self, config: UR5eConfiguration):
        self.config = config
        self.is_empty = False

    def empty(self) -> bool:
        return self.is_empty

    def reset(self) -> None:
        self.is_empty = False

    def grasp(self, grasping_point: GraspingPoint) -> bool:
        self.is_empty = True
        return True

    def workspace(self) -> GraspingWorkspace:
        return GraspingWorkspace([
            [self.config.reference_point[0] - self.config.workspace_size / 2, self.config.reference_point[0] + self.config.workspace_size / 2],
            [self.config.reference_point[1] - self.config.workspace_size / 2, self.config.reference_point[1] + self.config.workspace_size / 2],
            [self.config.reference_point[2], self.config.reference_point[2] + self.config.z_range]
        ])
