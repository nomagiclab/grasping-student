import copy

import cv2
import numpy as np
import time
import logging
import math

from dataset.schema import GraspingPoint
from robot import robotiq_gripper
from robot.abstract import UR5e, UR5eConfiguration

# this is the position right above the wall between two large boxes. Every complex move should start and end here.
from robot.robotiq_gripper_control import RobotiqGripper
from robot.robotiq_preamble import ROBOTIQ_PREAMBLE

safe = [0.52, 0., 0.22, 2.22, -2.22, 0.]
# the same position, symmetry for resetting the right box
qsafe_right = [0.2605343759059906, -1.6019445858397425, -1.652246356010437, -1.4609368753484269, 1.5790460109710693, 0.2598802149295807]
qsafe_left = [-3.4000662008868616, -1.052019552593567, 1.017395321522848, 1.6055502134510498, 1.565156102180481, 2.8823089599609375]
#TODO
qsafe_left = [-3.4008916060077112, -1.0520313543132325, 1.0173400084124964, 1.6058942514606933, 1.5639376640319824, -0.2603538672076624]

# the position in which gipper is on the boxes walls and it will calibrate them when closed
boxes_calibration = safe[:]
boxes_calibration[2] -= 0.05

above_right_box = [0.27, -0.25, 0.23, 0., 3.141, 0.]
gripping_right_box = above_right_box[:]
gripping_right_box[2] -= 0.06

above_left_box = [0.27, 0.26, 0.23, 0, 3.141, 0.]
#TODO
above_left_box = [0.27, 0.26, 0.23, 0, -3.141, 0.]
gripping_left_box = above_left_box[:]
gripping_left_box[2] -= 0.06

reset_right_checkpoints = [
        [0.27, -0.3, 0.56, 0., 3.141, 0.],
        [0.37607519072871703, 0.0020863602340042697, 0.5625790180321071, 0.7147534306637254, 2.232967412020244, -1.3149304756799407],
        [0.37608508487948655, 0.0954156274505257, 0.5625801070165717, 0.3863198961333237, 1.4476184931855667, -2.0145449855795134],
        [0.25718989315948376, 0.1252925961971147, 0.5625647851407708, 0.13261319822911674, 0.6825495401004007, -2.3063938919124487]]

reset_right_checkpoints = [
        #[0.26999525937196833, -0.27000162321082166, 0.5348730556026916, -2.479233128897897e-05, 3.1409758137272377, -1.2738872463274671e-05],
        #[0.26999525937196833, -0.28000162321082166, 0.5348730556026916, -2.479233128897897e-05, 3.1409758137272377, -1.2738872463274671e-05],
        [0.26999525937196833, -0.29000162321082166, 0.5348730556026916, -2.479233128897897e-05, 3.1409758137272377, -1.2738872463274671e-05],
        [0.46183209315445745, -0.03631190191665967, 0.597496545865811, 0.44381123121021576, 2.166698725941977, -1.0440429592238938],
        [0.544457215265994, 0.045838495529117396, 0.6287516566623392, 0.0037567137840812384, 1.4598127868599837, -1.5125028367516296],
        [0.5502605376578364, 0.02642217106770897, 0.7379097514186607, 0.15870122795856584, 0.6836717028936332, -1.5196448495461579],
        [0.5502845088201949, 0.026416124207921762, 0.7378945847479773, 0.2874676882835526, 0.5447353596883059, -1.4862547443250396] # new [maybe the one before can be skipped]
        ]



reset_left_checkpoints = [
            [0.27001121600226075, 0.28616117253483936, 0.471866061197124, -0.00011175028006181843, 3.1410060002725544, 1.448117303760917e-05],
            [0.2700089637491933, 0.18336333813626277, 0.5529633056258283, 6.681489095687025e-05, 3.074939712449576, 0.6417069107464598],
            [0.26997635784618135, 0.10867305432745508, 0.6119010833914638, 0.00018230284555980017, 2.9451129512952643, 1.0920958944863135],
            [0.3869989291475065, -0.053304646039729564, 0.5616147431052101, -0.31282924397879425, 1.838032343602827, 1.8435716985955066],
            [0.37150570935136845, -0.15061088193873415, 0.5616083953049436, -0.4328853434855269, 0.6511373333222729, 2.0137772984373252],
            ]

# the first [-2.6622939745532435, -1.0712129336646576, 0.25765973726381475, 2.383944197291992, 1.5651341676712036, -1.0981414953814905]
reset_left_checkpoints = [
        [0.27001892249892684, 0.29134236948099324, 0.49576952087179715, -9.593682815620005e-05, -3.1410161439776023, 2.2108609264685573e-05],
        [0.41284975628807985, -0.04217379394419122, 0.4972131667974079, 0.00023920956164402306, 2.7194562565986207, -2.4468514901388115e-05],
        [0.5442717216918739, 0.011579516903630148, 0.4972233907402013, 0.2694496530635816, 2.2018909431599027, 0.5712224340935008],
        [0.5442895737354541, 0.01154121410148766, 0.49724920920182797, 0.08523094138987482, 1.6386689540428554, 0.5648407835631395],
        [0.5526828363750546, 0.011573690600035394, 0.4972147921833022, 0.05571970942795778, 1.0357794092172512, 0.7799469445498048],
        [0.5526838991735931, 0.01155495589625527, 0.4971971470765458, -0.07679865789347141, 0.7086870393551453, 0.7582193671459382] # new [maybe the one before can be skipped]
        ]





# small boxes

above_small_box_right_down = [0.36655475992790637, -0.27, 0.2335044129821178, -0.00014248255313818663, 3.1410537951860085, 1.598555926783059e-05]
above_small_box_right_down_high = above_small_box_right_down[:]
above_small_box_right_down_high[2] += 0.32

gripping_small_box_right_down = above_small_box_right_down[:]
gripping_small_box_right_down[2] -= 0.13

above_small_box_right_left = [0.49921439680944657, -0.10154841087587309, 0.23839542157702807, -2.221053419060689, 2.221219002183467, -0.0004630419679182502]
gripping_small_box_right_left = above_small_box_right_left[:]
gripping_small_box_right_left[2] -= 0.13

above_small_box_inside_large_left_small_down = [0.3665570368021815, 0.26, 0.23, 0., 3.141, 0.]
above_small_box_inside_large_left_small_down_high = above_small_box_inside_large_left_small_down[:]
above_small_box_inside_large_left_small_down_high[2] += 0.32

gripping_small_box_inside_large_left_small_down = above_small_box_inside_large_left_small_down[:]
gripping_small_box_inside_large_left_small_down[2] -= 0.13

class UR5eTwoFinger(UR5e):
    def __init__(self, config: UR5eConfiguration):
        super().__init__(config)
        if config.initialize:
            self.gripper = RobotiqGripper(self.rtde_ctl)
            self.gripper.activate()  # returns to previous position after activation
            self.gripper.set_force(config.gripper_force)  # in percentage
            self.gripper.set_speed(config.gripper_speed)  # in percentage

            self.gripper_info = robotiq_gripper.RobotiqGripper()
            self.gripper_info.connect(self.config.robot_ip, 63352)

            self.rtde_ctl.zeroFtSensor()
            self.open_joints()

        if not self.config.skip_calibration and not self.config.skip_reset_bin_at_first_grasp:
            self.calibrate()


    def moveL(self, v):
        self.rtde_ctl.moveL(v, speed=0.3)

    def moveJ(self, v):
        self.rtde_ctl.moveJ(v, speed=0.3)


    def calibrate_middle(self):
        self.moveL(safe)
        self.gripper.open()
        self.moveL(boxes_calibration)
        self.gripper.close()
        self.gripper.open()
        self.moveL(safe)

    def calibrate_left(self):
        self.gripper.open()
        self.rtde_ctl.moveL(above_left_box)
        self.rtde_ctl.moveL(gripping_left_box)
        self.gripper.close()
        self.gripper.open()
        self.rtde_ctl.moveL(above_left_box)

    def calibrate_right(self):
        self.gripper.open()
        self.rtde_ctl.moveL(above_right_box)
        self.rtde_ctl.moveL(gripping_right_box)
        self.gripper.close()
        self.gripper.open()
        self.rtde_ctl.moveL(above_right_box)

    def reset_right(self):
        self.rtde_ctl.moveJ(qsafe_right)
        self.rtde_ctl.moveL(above_right_box)
        self.gripper.open()
        self.rtde_ctl.moveL(gripping_right_box)
        self.gripper.close()

        for checkpoint in reset_right_checkpoints + reset_right_checkpoints[::-1]:
            #self.rtde_ctl.moveL(checkpoint, speed=0.5, acceleration=0.2)
            self.rtde_ctl.moveL(checkpoint, speed=0.4, acceleration=0.1)
        self.moveL(gripping_right_box)
        self.gripper.open()
        self.rtde_ctl.moveL(above_right_box)
        self.rtde_ctl.moveL(safe)

    def reset_left(self):
        self.rtde_ctl.moveJ(qsafe_left)
        self.rtde_ctl.moveL(above_left_box)
        self.gripper.open()
        self.rtde_ctl.moveL(gripping_left_box)
        self.gripper.close()

        for checkpoint in reset_left_checkpoints + reset_left_checkpoints[::-1]:
            #self.rtde_ctl.moveL(checkpoint, speed=0.5, acceleration=0.2)
            self.rtde_ctl.moveL(checkpoint, speed=0.4, acceleration=0.1)

        self.moveL(gripping_left_box)
        self.gripper.open()
        self.rtde_ctl.moveL(above_left_box)
        self.rtde_ctl.moveL(safe)

    def calibrate_small(self, down=True, left=True):
        if self.config.skip_calibration:
            return

        if down:
            self.moveL(above_small_box_right_down)
            self.gripper.open()
            self.moveL(gripping_small_box_right_down)
            self.gripper.close()
            self.gripper.open()
            self.moveL(above_small_box_right_down)

        if left:
            self.moveL(above_small_box_right_left)
            self.gripper.open()
            self.moveL(gripping_small_box_right_left)
            self.gripper.close()
            self.gripper.open()
            self.moveL(above_small_box_right_left)

    def reset_small_right_left(self, difficulty=0.3):
        self.rtde_ctl.moveJ(qsafe_right)
        self.moveL(above_small_box_right_down)
        self.gripper.open()
        self.moveL(gripping_small_box_right_down)
        self.gripper.close()
        # set difficulty here
        self.rtde_ctl.moveL(above_small_box_right_down_high, acceleration=3., speed=difficulty)
        self.moveL(above_small_box_inside_large_left_small_down_high)
        self.rtde_ctl.moveL(gripping_small_box_inside_large_left_small_down, speed=0.15, acceleration=0.1)
        self.gripper.open()
        self.moveL(above_small_box_inside_large_left_small_down)

    def reset_small_left_right(self, difficulty=0.3):
        self.rtde_ctl.moveJ(qsafe_right)
        self.moveL(above_small_box_inside_large_left_small_down)
        self.gripper.open()
        self.moveL(gripping_small_box_inside_large_left_small_down)
        self.gripper.close()
        #set difficulty here
        self.rtde_ctl.moveL(above_small_box_inside_large_left_small_down_high, acceleration=3., speed=difficulty)
        self.moveL(above_small_box_right_down_high)
        self.rtde_ctl.moveL(gripping_small_box_right_down, speed=0.15, acceleration=0.1)
        self.gripper.open()
        self.moveL(above_small_box_right_down)

    def calibrate(self):
        if not self.config.skip_calibration:
            self.rtde_ctl.moveJ(qsafe_right)
            self.calibrate_middle()
            self.calibrate_left()
            self.calibrate_right()
            self.calibrate_small()

    def reset_boxes(self):
        self.reset_left()
        self.reset_small_right_left()
        self.reset_right()
        self.reset_small_left_right()

        if not self.config.skip_calibration:
            self.calibrate_middle()
            self.calibrate_small(left=True, down=False)

    def release(self):
        self.open_joints()

    def open_joints(self):
        self.gripper.set_position(self.config.opened_gripper_position)

    def reset_bin_two_boxes(self):
        del self.gripper
        del self.gripper_info
        self.gripper = RobotiqGripper(self.rtde_ctl)
        self.gripper.activate()  # returns to previous position after activation
        self.gripper.set_force(100)  # in percentage
        self.gripper.set_speed(100)  # in percentage
        self.gripper_info = robotiq_gripper.RobotiqGripper()
        self.gripper_info.connect(self.config.robot_ip, 63352)
        self.rtde_ctl.zeroFtSensor()

        if not self.config.reset_bin:
            self.open_joints()
            return
        self.reset_boxes()

    def reset_bin(self):
        del self.gripper
        del self.gripper_info
        self.gripper = RobotiqGripper(self.rtde_ctl)
        self.gripper.activate()  # returns to previous position after activation
        self.gripper.set_force(100)  # in percentage
        self.gripper.set_speed(100)  # in percentage
        self.gripper_info = robotiq_gripper.RobotiqGripper()
        self.gripper_info.connect(self.config.robot_ip, 63352)
        self.rtde_ctl.zeroFtSensor()

        if not self.config.reset_bin:
            self.open_joints()
            return


        bin_pos = copy.copy(self.config.reset_bin_position)
        bin_pos[2] += 0.2
        self.move(bin_pos)
        bin_pos[2] -= 0.21
        self.open_joints()
        self.move(bin_pos)
        self.gripper.close()
        bin_pos[0] = (self.workspace().limits()[0][0] + self.workspace().limits()[0][1]) / 2  # x
        bin_pos[1] = self.workspace().limits()[1][1] - (self.workspace().height() - self.config.bin_height) / 2  # y
        self.rtde_ctl.moveL(bin_pos, speed=self.config.speed / 2)

        bin_pos[2] += 0.3
        self.move(bin_pos, speed=self.config.speed * 2, acceleration=self.config.acceleration * 2)
        bin_pos[:2] = self.config.reset_bin_position[:2]
        self.move(bin_pos)
        self.move(self.config.reset_bin_position)
        self.open_joints()

        bin_pos = copy.copy(self.config.reset_bin_position)
        bin_pos[2] += 0.2
        self.move(bin_pos)
        self.move_home()

    def close_until_force(self, move_up=True, timeout=1.):
        self.gripper.close_async()

        begin = time.time()
        while time.time() < begin + timeout:
            if self.unsafe_rtde_ctl.toolContact([0., 0., 0., 0., 0., 0.]) >= {self.config.iters_tool_contact}:
                if move_up:
                    actual = self.rtde_info.getActualTCPPose()
                    actual[2] += 0.05
                    self.move(actual)
                    time.sleep(0.05)
                logging.info("Stopping, force on tcp detected.")
                return True

    def close_until_force_urscript(self, move_up=True, move_up_after_timeout=False, iters=100):
        self.rtde_ctl.sendCustomScriptFunction("WAIT_UNTIL_FORCE", ROBOTIQ_PREAMBLE + f"""
rq_set_force_norm({self.config.gripper_force})
rq_close()
iter = 0
while iter < {iters}:
    if tool_contact(p[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) >= {self.config.iters_tool_contact}:
        if {move_up}:
            actual = get_actual_tcp_pose()
            actual[2] = actual[2] + 0.05
            movel(actual)
        end
        break
    end
    iter = iter + 1
    sync()
end

if {move_up_after_timeout}:
    actual = get_actual_tcp_pose()
    actual[2] = actual[2] + 0.05
    movel(actual)
end
""")

    def grasp(self, grasping_point: GraspingPoint) -> bool:
        #self.rtde_ctl.moveL([0.5199719003476562, 0.2, 0.21994257070976916, 2.2199074925245057, -2.2199071955744807, -7.194608983234057e-05])
        self.rtde_ctl.moveJ([0.6095927953720093, -1.6920858822264613, -1.5547045469284058, -1.4687000227025528, 1.5796650648117065, 2.1457672119140625e-06])
        theta = math.pi - grasping_point.angle
        depth = grasping_point.z

        if grasping_point.z < 0.007 and self.config.skip_grasps_with_low_depth:
            return False

        # self.rtde_ctl.zeroFtSensor()

        pose = self.rtde_info.getActualTCPPose()
        pose[0] = grasping_point.x
        pose[1] = grasping_point.y
        pose[2] = self.workspace().limits()[2][1]

        initial = np.array([
            [0, -1, 0.],
            [-1, 0, 0],
            [0, 0, -1],
        ])
        rot_mat = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta),  0],
            [0,               0,                1],
        ])
        rot_mat = initial.dot(rot_mat)

        rvec, _ = cv2.Rodrigues(rot_mat)
        rvec = rvec.reshape(3)

        pose[3:] = rvec
        self.move(pose)

        offset = 0.015
        # offset to give some room to grasp,
        # not too big, to be able to grasp higher objects in clutter without collisions.
        pose[2] = self.workspace().limits()[2][0] + (max(0., depth - offset) if depth else 0)
        self.move_guarded_urscript(pose, speed=min(self.config.speed / 5., 0.05), acceleration=1)
        self.close_until_force_urscript(move_up=True, move_up_after_timeout=True, iters=1000)

        gripper_pos = self.gripper_info.get_current_position()
        successful = self.gripper.detect_object() and not self.gripper_info.is_closed() and gripper_pos < 220

        if successful:
            self.gripper.set_position(gripper_pos)
            self.close_until_force_urscript(move_up=False)
            gripper_pos_bis = self.gripper_info.get_current_position()
            successful = successful and self.gripper.detect_object()\
                         and abs(gripper_pos - gripper_pos_bis) < self.config.max_finger_diff_object_detection \
                         and not self.gripper_info.is_closed() and gripper_pos_bis < 220 and gripper_pos < 220
            time.sleep(0.2)  # Strange rtde control script is not running otherwise

        #self.rtde_ctl.moveL([0.5199719003476562, 0.2, 0.21994257070976916, 2.2199074925245057, -2.2199071955744807, -7.194608983234057e-05])
        self.rtde_ctl.moveJ([0.6095927953720093, -1.6920858822264613, -1.5547045469284058, -1.4687000227025528, 1.5796650648117065, 2.1457672119140625e-06])
        if successful:
            self.place_to_bin()
        else:
            self.open_joints()

        self.move_home()
        return successful
