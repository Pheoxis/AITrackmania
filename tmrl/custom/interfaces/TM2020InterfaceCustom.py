import cv2
import numpy as np
from gymnasium import spaces

from custom.interfaces.TM2020Interface import TM2020Interface
from custom.utils.control_mouse import mouse_save_replay_tm20

import config.config_constants as cfg


class TM2020InterfaceCustom(TM2020Interface):
    def __init__(
            self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(160),
            record=False, save_replay: bool = False,
            grayscale: bool = False, resize_to: tuple = (128, 64),
            finish_reward=cfg.REWARD_END_OF_TRACK, constant_penalty: float = 0.25,
            crash_penalty=cfg.CRASH_PENALTY
    ):
        super().__init__(
            img_hist_len=img_hist_len, gamepad=gamepad, min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replay, grayscale=grayscale, finish_reward=finish_reward, resize_to=resize_to,
            constant_penalty=constant_penalty, crash_penalty=crash_penalty
        )
        self.record = record
        self.window_interface = None

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

        steer_angle = spaces.Box(low=-1000.0, high=1000.0, shape=(2,))  # fl, fr

        wheel_rot = spaces.Box(low=0.0, high=1700.0, shape=(2,))  # fl, fr

        wheel_rot_speed = spaces.Box(low=-1000.0, high=1000.0, shape=(2,))  # fl, fr

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

        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(h, w, 3))  # cv2 images are (h, w, c)

        return spaces.Tuple(
            (
                speed,
                race_progress,
                input_steer, input_gas_pedal, input_brake,
                acceleration, jerk,
                rpm,
                aim_yaw, aim_pitch,
                steer_angle,
                wheel_rot,
                wheel_rot_speed,
                damper_len,  # 4
                slip_coef,  # 4
                reactor_air_control,  # 3
                ground_dist,
                crashed,
                reactor_ground_mode,
                ground_contact,
                gear,
                surface_id,  # 4
                failure_counter,
                img
            )
        )

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.grab_data()
        # print(f"data: {data}")
        self.img = img  # for render()
        return data, img

    def grab_data(self):
        data = self.client.retrieve_data()
        return data

    def get_obs_rew_terminated_info(self):
        """
            returns the observation, the reward, and a terminated signal for end of episode
            obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        # print(f"data: {data}")

        speed = np.array([data[0]], dtype='float32')
        acceleration = np.array([data[9]], dtype='float32')
        jerk = np.array([data[10]], dtype='float32')

        race_progress = np.array([data[1]], dtype='float32')

        input_steer = np.array([data[5]], dtype='float32')
        input_gas_pedal = np.array([data[6]], dtype='float32')

        rpm = np.array([data[11]], dtype='float32')

        aim_yaw = np.array([data[12]], dtype='float32')
        aim_pitch = np.array([data[13]], dtype='float32')

        steer_angle = np.array([data[14:16]], dtype='float32')
        wheel_rot = np.array([data[16:18]], dtype='float32')
        wheel_rot_speed = np.array([data[18:20]], dtype='float32')
        damper_len = np.array([data[20:24]], dtype='float32')
        slip_coef = np.array([data[24:28]], dtype='float32')
        reactor_air_control = np.array([data[28:31]], dtype='float32')
        ground_dist = np.array([data[31]], dtype='float32')

        input_brake = np.array([data[7]], dtype='float32')
        reactor_ground_mode = np.array([data[33]], dtype='float32')
        ground_contact = np.array([data[34]], dtype='float32')

        gear = np.array([data[35]], dtype='float32')

        surface_id = np.array([data[36:40]], dtype='float32')

        crashed = np.array([data[32]], dtype='int16')
        end_of_track = bool(data[8])

        info = {}
        reward = 0.0
        if end_of_track:
            terminated = True
            reward += self.finish_reward
            failure_counter = 0
            if self.save_replays:
                mouse_save_replay_tm20(True)
        else:
            rew, terminated, failure_counter = self.reward_function.compute_reward(
                pos=np.array([data[2], data[3], data[4]]),  # position x,y,z
                crashed=bool(crashed), gas_input=bool(input_gas_pedal), brake_input=bool(input_brake),
                speed=speed[0]
            )
            reward += rew

        failure_counter = float(failure_counter)
        img = np.array(img)

        observation = [
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear, rpm,
            aim_yaw, aim_pitch,
            surface_id, steer_angle, wheel_rot, wheel_rot_speed, damper_len, slip_coef,
            reactor_ground_mode, ground_contact, reactor_air_control, ground_dist,
            crashed, failure_counter,
            img
        ]

        reward = np.float32(reward)
        # print(f"Reward: {reward}, crashed {bool(crashed)}, race progress {round(race_progress[0], 2)}")
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()

        speed = np.array([data[0]], dtype='float32')
        acceleration = np.array([data[9]], dtype='float32')
        jerk = np.array([data[10]], dtype='float32')

        race_progress = np.array([data[1]], dtype='float32')

        input_steer = np.array([data[5]], dtype='float32')
        input_gas_pedal = np.array([data[6]], dtype='float32')

        rpm = np.array([data[11]], dtype='float32')

        aim_yaw = np.array([data[12]], dtype='float32')
        aim_pitch = np.array([data[13]], dtype='float32')

        steer_angle = np.array([data[14:16]], dtype='float32')
        wheel_rot = np.array([data[16:18]], dtype='float32')
        wheel_rot_speed = np.array([data[18:20]], dtype='float32')
        damper_len = np.array([data[20:24]], dtype='float32')
        slip_coef = np.array([data[24:28]], dtype='float32')
        reactor_air_control = np.array([data[28:31]], dtype='float32')
        ground_dist = np.array([data[31]], dtype='float32')

        input_brake = np.array([data[7]], dtype='float32')
        reactor_ground_mode = np.array([data[33]], dtype='float32')
        ground_contact = np.array([data[34]], dtype='float32')

        gear = np.array([data[35]], dtype='float32')

        surface_id = np.array([data[36:40]], dtype='float32')

        crashed = np.array([data[32]], dtype='int16')

        failure_counter = 0.0

        img = np.array(img)

        observation = [
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear, rpm,
            aim_yaw, aim_pitch,
            surface_id, steer_angle, wheel_rot, wheel_rot_speed, damper_len, slip_coef,
            reactor_ground_mode, ground_contact, reactor_air_control, ground_dist,
            crashed, failure_counter,
            img
        ]

        self.reward_function.reset()
        return observation, {}
