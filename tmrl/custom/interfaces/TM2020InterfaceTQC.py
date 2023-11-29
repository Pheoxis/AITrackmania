import cv2
import numpy as np
from gymnasium import spaces

from custom.interfaces.TM2020Interface import TM2020Interface
from custom.utils.control_mouse import mouse_save_replay_tm20

import config.config_constants as cfg


class TM2020InterfaceTQC(TM2020Interface):
    def __init__(
            self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(160),
            record=False, save_replay: bool = False,
            grayscale: bool = False, resize_to: tuple = (128, 64),
            finish_reward=cfg.END_OF_TRACK_REWARD, constant_penalty: float = 0.05,
            crash_penalty=cfg.CRASH_PENALTY, checkpoint_reward=cfg.CHECKPOINT_REWARD,
            lap_reward=cfg.LAP_REWARD
    ):
        super().__init__(
            img_hist_len=img_hist_len, gamepad=gamepad, min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replay, grayscale=grayscale, finish_reward=finish_reward, resize_to=resize_to,
            constant_penalty=constant_penalty, crash_penalty=crash_penalty
        )
        self.record = record
        self.window_interface = None
        self.cur_lap = 0
        self.cur_checkpoint = 0
        self.lap_reward = lap_reward
        self.checkpoint_reward = checkpoint_reward

    def get_observation_space(self):
        # https://gymnasium.farama.org/api/spaces/
        """Returns the observation space.

            Returns:
                observation_space: gymnasium.spaces.Tuple

            Note: Do NOT put the action buffer here (automated).
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))

        distance = spaces.Box(low=-100.0, high=50_000.0, shape=(1,))

        pos = spaces.Box(low=-10000.0, high=10000.0, shape=(3,))

        input_steer = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_gas_pedal = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_brake = spaces.Box(low=0.0, high=1.0, shape=(1,))

        acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
        jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))

        rpm = spaces.Box(low=0.0, high=20_000.0, shape=(1,))
        aim_yaw = spaces.Box(low=-4.0, high=4.0, shape=(1,))
        aim_pitch = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        race_progress = spaces.Box(low=0.0, high=1_000_000, shape=(1,))

        steer_angle = spaces.Box(low=-1000.0, high=1000.0, shape=(2,))  # fl, fr

        slip_coef = spaces.Box(low=0.0, high=1.0, shape=(2,))  # fl, fr

        crashed = spaces.Box(low=0.0, high=1.0, shape=(1,))

        gear = spaces.Box(low=0.0, high=6.0, shape=(1,))

        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))

        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            imgs = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            imgs = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)

        return spaces.Tuple(
            (
                pos, distance,
                speed, acceleration, jerk,
                race_progress,
                input_steer, input_gas_pedal, input_brake,
                gear, rpm,
                aim_yaw, aim_pitch,
                steer_angle, slip_coef,
                crashed, failure_counter,
                imgs
            )
        )

    def grab_data_and_img(self, percentage_to_cut: float = 0.2):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        height, _ = img.shape[:2]
        cut_height = int(height * percentage_to_cut)
        cut_timer_height = int(height * 0.95)
        img = img[cut_height:cut_timer_height, :]
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.grab_data()
        # print(f"data: {data}")
        self.img = img  # for render()
        # cv2.imshow("Environment", img)
        # cv2.waitKey(1)
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
        curCP = int(data[0])
        curLap = int(data[1])

        speed = np.array([data[2]], dtype='float32')

        distance = np.array([data[2]], dtype='float32')

        pos = np.array([data[4], data[5], data[6]], dtype='float32')

        input_steer = np.array([data[7]], dtype='float32')
        input_gas_pedal = np.array([data[8]], dtype='float32')
        input_brake = np.array([data[9]], dtype='float32')

        acceleration = np.array([data[11]], dtype='float32')
        jerk = np.array([data[12]], dtype='float32')

        rpm = np.array([data[13]], dtype='float32')

        aim_yaw = np.array([data[14]], dtype='float32')
        aim_pitch = np.array([data[15]], dtype='float32')

        steer_angle = np.array([data[16:18]], dtype='float32')

        slip_coef = np.array([data[18:20]], dtype='float32')

        crashed = np.array([data[20]], dtype='float32')

        gear = np.array([data[21]], dtype='float32')

        rew, terminated, failure_counter, race_progress, reward_sum = self.reward_function.compute_reward(
            pos=pos,  # position x,y,z
            crashed=bool(crashed),  # distance=bool(input_gas_pedal), brake_input=bool(input_brake),
            speed=speed[0]
        )

        end_of_track = bool(data[10])

        if end_of_track:
            terminated = True  # sprawdzić czy to wgl jest wywoływane
            rew += self.finish_reward
            reward_sum += self.finish_reward
            failure_counter = 0.0
            if self.save_replays:
                mouse_save_replay_tm20(True)

        if self.cur_lap < curLap:
            rew += self.lap_reward
            self.cur_lap = curLap

        if self.cur_checkpoint < curCP:
            rew += self.checkpoint_reward
            self.cur_checkpoint = curCP

        race_progress = np.array([race_progress], dtype='float32')

        failure_counter = float(failure_counter)
        self.img_hist.append(img)
        imgs = np.array(self.img_hist)
        info = {"reward_sum": reward_sum}
        observation = [
            pos, distance,
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear, rpm,
            aim_yaw, aim_pitch,
            steer_angle, slip_coef,
            crashed, failure_counter,
            imgs
        ]

        reward = np.float32(rew)
        # print(f"Reward: {reward}, crashed {bool(crashed)}, race progress {round(race_progress[0], 2)}")
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()

        self.cur_lap = 0
        self.cur_checkpoint = 0

        speed = np.array([data[2]], dtype='float32')

        distance = np.array([data[2]], dtype='float32')

        pos = np.array([data[4], data[5], data[6]], dtype='float32')

        input_steer = np.array([data[7]], dtype='float32')
        input_gas_pedal = np.array([data[8]], dtype='float32')
        input_brake = np.array([data[9]], dtype='float32')

        acceleration = np.array([data[11]], dtype='float32')
        jerk = np.array([data[12]], dtype='float32')

        rpm = np.array([data[13]], dtype='float32')

        aim_yaw = np.array([data[14]], dtype='float32')
        aim_pitch = np.array([data[15]], dtype='float32')

        steer_angle = np.array([data[16:18]], dtype='float32')

        slip_coef = np.array([data[18:20]], dtype='float32')

        crashed = np.array([data[20]], dtype='float32')

        gear = np.array([data[21]], dtype='float32')

        failure_counter = 0.0
        race_progress = 0.0

        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        observation = [
            pos, distance,
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear, rpm,
            aim_yaw, aim_pitch,
            steer_angle, slip_coef,
            crashed, failure_counter,
            imgs
        ]

        self.reward_function.reset()
        info = {"reward_sum": 0.0}
        return observation, info
