import numpy as np
from gymnasium import spaces

from custom.interfaces.TM2020Interface import TM2020Interface
from custom.utils.control_mouse import mouse_save_replay_tm20

import config.config_constants as cfg


class TM2020InterfaceTQCmini(TM2020Interface):
    def __init__(
            self, img_hist_len=1, gamepad=False, min_nb_steps_before_failure=int(160),
            record=False, save_replay: bool = False,
            grayscale: bool = False, resize_to: tuple = (128, 64),
            finish_reward=cfg.REWARD_END_OF_TRACK, constant_penalty: float = 0.05,
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

        input_steer = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_gas_pedal = spaces.Box(low=-1000.0, high=1000.0, shape=(1,))
        input_brake = spaces.Box(low=0.0, high=1.0, shape=(1,))

        acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
        jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))

        aim_yaw = spaces.Box(low=-4.0, high=4.0, shape=(1,))
        aim_pitch = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        race_progress = spaces.Box(low=0.0, high=1_000_000, shape=(1,))

        steer_angle = spaces.Box(low=-1000.0, high=1000.0, shape=(2,))  # fl, fr

        slip_coef = spaces.Box(low=0.0, high=1.0, shape=(2,))  # fl, fr

        crashed = spaces.Box(low=0.0, high=1.0, shape=(1,))

        gear = spaces.Box(low=0.0, high=6.0, shape=(1,))

        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))

        next_checkpoints = spaces.Box(low=-100.0, high=100.0, shape=(2*cfg.POINTS_NUMBER,))

        return spaces.Tuple(
            (
                next_checkpoints,
                speed, acceleration, jerk,
                race_progress,
                input_steer, input_gas_pedal, input_brake,
                gear,
                aim_yaw, aim_pitch,
                steer_angle, slip_coef,
                crashed, failure_counter
            )
        )

    # def grab_data_and_img(self, percentage_to_cut: float = 0.2):
    #     img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
    #     height, _ = img.shape[:2]
    #     cut_height = int(height * percentage_to_cut)
    #     img = img[cut_height:, :]
    #     if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
    #         img = cv2.resize(img, self.resize_to)
    #     if self.grayscale:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         img = img[:, :, ::-1]  # reversed view for numpy RGB convention
    #     data = self.grab_data()
    #     # print(f"data: {data}")
    #     self.img = img  # for render()
    #     # cv2.imshow("Environment", img)
    #     # cv2.waitKey(1)
    #     return data, img

    def grab_data(self):
        data = self.client.retrieve_data()
        return data

    def get_obs_rew_terminated_info(self):
        """
            returns the observation, the reward, and a terminated signal for end of episode
            obs must be a list of numpy arrays
        """
        data = self.grab_data()
        # print(f"data: {data}")
        cur_cp = int(data[0])
        cur_lap = int(data[1])

        speed = np.array([data[2]], dtype='float32')

        pos = np.array([data[3], data[4], data[5]], dtype='float32')

        input_steer = np.array([data[6]], dtype='float32')
        input_gas_pedal = np.array([data[7]], dtype='float32')
        input_brake = np.array([data[8]], dtype='float32')

        acceleration = np.array([data[10]], dtype='float32')
        jerk = np.array([data[11]], dtype='float32')

        aim_yaw = np.array([data[12]], dtype='float32')
        aim_pitch = np.array([data[13]], dtype='float32')

        steer_angle = np.array(data[14:16], dtype='float32')

        slip_coef = np.array(data[16:18], dtype='float32')

        crashed = np.array([data[18]], dtype='float32')

        gear = np.array([data[19]], dtype='float32')

        rew, terminated, failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=pos,  # position x,y,z
            crashed=bool(crashed),
            speed=speed[0]
        )

        race_progress = self.reward_function.compute_race_progress()

        next_checkpoints = self.reward_function.get_n_next_checkpoints_xy(pos, cfg.POINTS_NUMBER)

        end_of_track = bool(data[9])

        if end_of_track:
            terminated = True  # sprawdzić czy to wgl jest wywoływane
            rew += self.finish_reward
            reward_sum += self.finish_reward
            failure_counter = 0.0
            if self.save_replays:
                mouse_save_replay_tm20(True)

        if self.cur_lap < cur_lap:
            rew += self.lap_reward
            self.cur_lap = cur_lap

        if self.cur_checkpoint < cur_cp:
            rew += self.checkpoint_reward
            self.cur_checkpoint = cur_cp

        race_progress = np.array([race_progress], dtype='float32')

        failure_counter = np.array([float(failure_counter)])
        info = {"reward_sum": reward_sum}

        observation = [
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear,
            aim_yaw, aim_pitch,
            steer_angle, slip_coef,
            crashed, failure_counter
        ]

        total_obs = [next_checkpoints] + observation

        total_obs[0] = np.array(total_obs[0])

        reward = np.float32(rew)
        # print(f"Reward: {reward}, crashed {bool(crashed)}, race progress {round(race_progress[0], 2)}")
        return total_obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data = self.grab_data()

        self.cur_lap = 0
        self.cur_checkpoint = 0

        speed = np.array([data[2]], dtype='float32')

        pos = np.array([data[3], data[4], data[5]], dtype='float32')

        input_steer = np.array([data[6]], dtype='float32')
        input_gas_pedal = np.array([data[7]], dtype='float32')
        input_brake = np.array([data[8]], dtype='float32')
        # isFinished 10
        acceleration = np.array([data[10]], dtype='float32')
        jerk = np.array([data[11]], dtype='float32')

        aim_yaw = np.array([data[12]], dtype='float32')
        aim_pitch = np.array([data[13]], dtype='float32')

        steer_angle = np.array(data[14:16], dtype='float32')

        slip_coef = np.array(data[16:18], dtype='float32')

        crashed = np.array([data[18]], dtype='float32')

        gear = np.array([data[19]], dtype='float32')

        failure_counter = np.array([0.0])
        race_progress = 0.0

        next_checkpoints = self.reward_function.get_n_next_checkpoints_xy(pos, cfg.POINTS_NUMBER)

        observation = [
            speed, acceleration, jerk,
            race_progress,
            input_steer, input_gas_pedal, input_brake,
            gear,
            aim_yaw, aim_pitch,
            steer_angle, slip_coef,
            crashed, failure_counter
        ]

        total_obs = [next_checkpoints] + observation

        total_obs[0] = np.array(total_obs[0])

        self.reward_function.reset()
        info = {"reward_sum": 0.0}
        return total_obs, info