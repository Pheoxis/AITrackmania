import numpy as np
from gymnasium import spaces

from custom.interfaces.TM2020Interface import TM2020Interface
from custom.utils.control_mouse import mouse_save_replay_tm20
from custom.utils.tools import TM2020OpenPlanetClient

import config.config_constants as cfg


class TM2020InterfaceCustom(TM2020Interface):
    def __init__(
            self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(20 * 3.5),
            record=False, save_replay: bool = False, crash_penalty: int = 10,
            grayscale: bool = False, resize_to: tuple = (128, 64)
    ):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_failure, save_replay, grayscale, resize_to)
        self.record = record
        self.window_interface = None
        self.crash_penalty = crash_penalty
        self.client = TM2020OpenPlanetClient()

    def get_observation_space(self):
        # https://gymnasium.farama.org/api/spaces/
        """Returns the observation space.

            Returns:
                observation_space: gymnasium.spaces.Tuple

            Note: Do NOT put the action buffer here (automated).
        """

        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
        jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))

        race_progress = spaces.Box(low=0.0, high=110.0, shape=(1,))

        input_steer = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_gas_pedal = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))

        rpm = spaces.Box(low=0.0, high=20_000.0, shape=(1,))
        aim_yaw = spaces.Box(low=-4.0, high=4.0, shape=(1,))
        aim_pitch = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        steer_angle = spaces.Box(low=-1000.0, high=1000.0, shape=(4,))  # fl, fr, rl, rr

        wheel_rot = spaces.Box(low=0.0, high=1700.0, shape=(4,))  # fl, fr, rl, rr

        wheel_rot_speed = spaces.Box(low=-1000.0, high=1000.0, shape=(4,))  # fl, fr, rl, rr

        damper_len = spaces.Box(low=0.0, high=0.1, shape=(4,))  # fl, fr, rl, rr

        slip_coef = spaces.Box(low=0.0, high=1.0, shape=(4,))  # fl, fr, rl, rr

        reactor_air_control = spaces.Box(low=-1000.0, high=1000.0, shape=(3,))  # xyz

        ground_dist = spaces.Box(low=0.0, high=100.0, shape=(1,))

        input_brake = spaces.Box(low=0.0, high=1.0, shape=(1,))
        crashed = spaces.Box(low=0.0, high=1.0, shape=(1,))

        reactor_ground_mode = spaces.Box(low=0.0, high=1.0, shape=(1,))
        ground_contact = spaces.Box(low=0.0, high=1.0, shape=(1,))

        gear = spaces.Box(low=0.0, high=6.0, shape=(1,))
        surface_id = spaces.Box(low=0.0, high=23.0, shape=(4,))  # fl, fr, rl, rr

        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))

        # input_brake = spaces.Discrete(2)
        # crashed = spaces.Discrete(2)
        #
        # reactor_ground_mode = spaces.Discrete(2)
        # ground_contact = spaces.Discrete(2)
        #
        # gear = spaces.Discrete(7)
        # surface_id = spaces.MultiDiscrete([23 for _ in range(4)])  # fl, fr, rl, rr

        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)

        return spaces.Tuple(
            (
                speed, gear, rpm,
                acceleration, jerk,
                race_progress,
                input_steer, input_gas_pedal, input_brake,
                aim_yaw, aim_pitch,
                surface_id, steer_angle, wheel_rot, wheel_rot_speed, damper_len, slip_coef,
                reactor_ground_mode, ground_contact, reactor_air_control, ground_dist,
                crashed, failure_counter,
                img
            )
        )
        # region code
        # speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        # acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
        # jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))
        # race_progress = spaces.Box(low=0.0, high=110.0, shape=(1,))
        # position_x = spaces.Box(low=0.0, high=110.0, shape=(1,))
        # position_y = spaces.Box(low=0.0, high=110.0, shape=(1,))
        # position_z = spaces.Box(low=0.0, high=110.0, shape=(1,))
        #
        # input_gas_pedal = spaces.Box(low=0.0, high=1.0, shape=(1,))
        # input_is_braking = spaces.Box(low=0.0, high=1.0, shape=(1,))
        #
        # gear = spaces.Box(low=0.0, high=6, shape=(1,))
        # rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        #
        # aim_yaw = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # aim_pitch = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        #
        # fl_surface_id = spaces.Box(low=0.0, high=23.0, shape=(1,))
        # fl_steer_angle = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        # fl_wheel_rot = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # fl_wheel_rot_speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        # fl_damper_len = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # fl_slip_coef = spaces.Box(low=0.0, high=1.0, shape=(1,))
        #
        # fr_surface_id = spaces.Box(low=0.0, high=23.0, shape=(1,))
        # fr_steer_angle = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        # fr_wheel_rot = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # fr_wheel_rot_speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        # fr_damper_len = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # fr_slip_coef = spaces.Box(low=0.0, high=1.0, shape=(1,))
        #
        # rl_surface_id = spaces.Box(low=0.0, high=23.0, shape=(1,))
        # rl_steer_angle = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        # rl_wheel_rot = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # rl_wheel_rot_speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        # rl_damper_len = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # rl_slip_coef = spaces.Box(low=0.0, high=1.0, shape=(1,))
        #
        # rr_surface_id = spaces.Box(low=0.0, high=23.0, shape=(1,))
        # rr_steer_angle = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        # rr_wheel_rot = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # rr_wheel_rot_speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        # rr_damper_len = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        # rr_slip_coef = spaces.Box(low=0.0, high=1.0, shape=(1,))
        #
        # reactor_ground_mode = spaces.Box(low=0, high=1, shape=(1,))
        # ground_contact = spaces.Box(low=0, high=1, shape=(1,))
        #
        # reactor_air_control_x = spaces.Box(low=0.0, high=110.0, shape=(1,))
        # reactor_air_control_y = spaces.Box(low=0.0, high=110.0, shape=(1,))
        # reactor_air_control_z = spaces.Box(low=0.0, high=110.0, shape=(1,))
        #
        # ground_dist = spaces.Box(low=0.0, high=110.0, shape=(1,))
        # crash = spaces.Box(low=0.0, high=1, shape=(1,))

        # return spaces.Tuple(
        #     (
        #         speed, acceleration, jerk,
        #         race_progress,
        #         position_x, position_y, position_z,
        #         input_gas_pedal, input_is_braking,
        #         gear, rpm,
        #         aim_yaw, aim_pitch,
        #         fl_surface_id, fl_steer_angle, fl_wheel_rot, fl_wheel_rot_speed, fl_damper_len, fl_slip_coef,
        #         fr_surface_id, fr_steer_angle, fr_wheel_rot, fr_wheel_rot_speed, fr_damper_len, fr_slip_coef,
        #         rl_surface_id, rl_steer_angle, rl_wheel_rot, rl_wheel_rot_speed, rl_damper_len, rl_slip_coef,
        #         rr_surface_id, rr_steer_angle, rr_wheel_rot, rr_wheel_rot_speed, rr_damper_len, rr_slip_coef,
        #         reactor_ground_mode,
        #         ground_contact,
        #         reactor_air_control_x, reactor_air_control_y, reactor_air_control_z,
        #         ground_dist,
        #         crash
        #     )
        # )
        # endregion code

    def get_obs_rew_terminated_info(self):
        """
            returns the observation, the reward, and a terminated signal for end of episode
            obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        # print(f"data: {data}")

        speed = np.array([data[0]], dtype='float32')
        acceleration = np.array([data[1]], dtype='float32')
        jerk = np.array([data[2]], dtype='float32')

        race_progress = np.array([data[3]], dtype='float32')

        input_steer = np.array([data[7]], dtype='float32')
        input_gas_pedal = np.array([data[8]], dtype='float32')

        rpm = np.array([data[9]], dtype='float32')

        aim_yaw = np.array([data[10]], dtype='float32')
        aim_pitch = np.array([data[11]], dtype='float32')

        steer_angle = np.array([data[12:13]], dtype='float32')
        wheel_rot = np.array([data[14:15]], dtype='float32')
        wheel_rot_speed = np.array([data[16:17]], dtype='float32')
        damper_len = np.array([data[18:21]], dtype='float32')
        slip_coef = np.array([data[22:25]], dtype='float32')
        reactor_air_control = np.array([data[26:28]], dtype='float32')
        ground_dist = np.array([data[29]], dtype='float32')

        input_brake = np.array([data[31]], dtype='float32')
        reactor_ground_mode = np.array([data[33]], dtype='float32')
        ground_contact = np.array([data[34]], dtype='float32')

        gear = np.array([data[35]], dtype='float32')

        surface_id = np.array([data[36:39]], dtype='float32')

        crashed = np.array([data[32]], dtype='float32')
        end_of_track = bool(data[30])

        info = {}
        reward = 0
        if crashed == 1:
            reward -= self.crash_penalty
        if end_of_track:
            terminated = True
            reward += self.finish_reward
            failure_counter = 0
            if self.save_replays:
                mouse_save_replay_tm20(True)
        else:
            rew, terminated, failure_counter = self.reward_function.compute_reward(
                pos=np.array([data[4], data[5], data[6]])  # position x,y,z
            )
            reward += rew

        failure_counter = float(failure_counter)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        observation = [
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear, rpm,
            aim_yaw, aim_pitch,
            surface_id, steer_angle, wheel_rot, wheel_rot_speed, damper_len, slip_coef,
            reactor_ground_mode, ground_contact, reactor_air_control, ground_dist,
            crashed, failure_counter,
            imgs
        ]

        reward += self.constant_penalty
        reward = np.float32(reward)
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()

        speed = np.array([data[0]], dtype='float32')
        acceleration = np.array([data[1]], dtype='float32')
        jerk = np.array([data[2]], dtype='float32')

        race_progress = np.array([data[3]], dtype='float32')

        input_steer = np.array([data[7]], dtype='float32')
        input_gas_pedal = np.array([data[8]], dtype='float32')

        rpm = np.array([data[9]], dtype='float32')

        aim_yaw = np.array([data[10]], dtype='float32')
        aim_pitch = np.array([data[11]], dtype='float32')

        steer_angle = np.array([data[12:13]], dtype='float32')
        wheel_rot = np.array([data[14:15]], dtype='float32')
        wheel_rot_speed = np.array([data[16:17]], dtype='float32')
        damper_len = np.array([data[18:21]], dtype='float32')
        slip_coef = np.array([data[22:25]], dtype='float32')
        reactor_air_control = np.array([data[26:28]], dtype='float32')
        ground_dist = np.array([data[29]], dtype='float32')

        input_brake = np.array([data[31]], dtype='float32')
        reactor_ground_mode = np.array([data[33]], dtype='float32')
        ground_contact = np.array([data[34]], dtype='float32')

        gear = np.array([data[35]], dtype='float32')

        surface_id = np.array([data[36:39]], dtype='float32')

        crashed = np.array(data[32], dtype='float32')

        failure_counter = 0.0

        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        observation = [
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear, rpm,
            aim_yaw, aim_pitch,
            surface_id, steer_angle, wheel_rot, wheel_rot_speed, damper_len, slip_coef,
            reactor_ground_mode, ground_contact, reactor_air_control, ground_dist,
            crashed, failure_counter,
            imgs
        ]

        self.reward_function.reset()
        return observation, {}
