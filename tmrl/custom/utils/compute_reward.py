# standard library imports
import atexit
import os
import pickle
import shutil
import tempfile
import time

# third-party imports
import numpy as np
import logging
from config import config_constants as cfg

import wandb

logging.basicConfig(level=logging.INFO)


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """

    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=12,
                 nb_obs_backward=12,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=int(2.5 * 20),
                 max_dist_from_traj=50.0,
                 crash_penalty=10.0,
                 constant_penalty=0.0,
                 low_threshold=10,
                 high_threshold=250):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (as a number of positions in the trajectory)
            nb_obs_backward: same thing but for when rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this number of steps with no reward, episode is terminated
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            max_dist_from_traj: the reward is 0 if the car is further than this distance from the demo trajectory
        """
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.cur_idx = 0
        self.prev_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)
        # self.survive_reward = 0.5
        self.crash_penalty = crash_penalty
        self.crash_counter = 1
        self.constant_penalty = constant_penalty
        # self.gas_reward = 0.0001
        # self.brake_penalty = -0.0002
        # self.counter = 0
        self.reward_sum = 0.0
        self.isFinished = False
        self.reward_sum_list = []
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.cur_distance = 0
        self.prev_distance = 0

        wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
        atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
        wandb_initialized = False
        err_cpt = 0
        while not wandb_initialized:
            try:
                wandb.init(project=cfg.WANDB_PROJECT, entity=cfg.WANDB_ENTITY, id=cfg.WANDB_RUN_ID + "_rewards")
                wandb_initialized = True
            except Exception as e:
                err_cpt += 1
                logging.warning(f"wandb error {err_cpt}: {e}")
                if err_cpt > 10:
                    logging.warning(f"Could not connect to wandb, aborting.")
                    exit()
                else:
                    time.sleep(10.0)
        # self.tmp_counter = 0
        # self.traj = []

    def compute_reward(self, pos, crashed: bool = False, distance: float = None, speed: float = None):
        # self.tmp_counter += 1
        # if self.tmp_counter % 10 == 0:
        #     print(f"pos: {pos}, first pos from reward: {self.data[0]}")
        terminated = False
        self.step_counter += 1
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = 0
        while True:
            dist = np.linalg.norm(pos - self.data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            # stop condition
            if index >= self.datalen or temp <= 0:
                break
        reward = (best_index - self.cur_idx) / 100.0
        if best_index == self.cur_idx:  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                # stop condition
                if index <= 0 or temp <= 0:
                    break
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True

        else:
            self.failure_counter = 0
        self.cur_idx = best_index

        # if reward > 0.0:
        # reward += speed * (1 + self.sigmoid(self.cur_idx - self.prev_idx)) / 200.0
        # reward += speed / 250.0
        # self.prev_idx = self.cur_idx
        # self.prev_distance = distance

        if not self.isFinished:
            if self.cur_idx > len(self.data) - 5:
                reward += cfg.REWARD_END_OF_TRACK
                self.isFinished = True

        if speed < 0:
            reward -= 1.0

        # if self.cur_idx > self.prev_idx:
        #     reward += speed * (distance - self.prev_distance) / (self.cur_idx - self.prev_idx)
        #     self.prev_idx = self.cur_idx
        #     self.prev_distance = distance

        if crashed:
            reward -= abs(self.crash_penalty) * self.crash_counter
            self.crash_counter += 1

        reward -= abs(self.constant_penalty)

        self.reward_sum += reward
        if terminated:
            # self.counter += 1
            print(f"Total reward of the run: {self.reward_sum}")
            self.isFinished = False
            if self.reward_sum != 0.0:
                self.reward_sum_list.append(self.reward_sum)
                wandb.log({"Run reward": self.reward_sum})
            # logging.info(f"Total reward of the run: {self.reward_sum}")
            # if self.counter % 2 == 0:
            #     self.min_nb_steps_before_failure += 2

        return reward, terminated, self.failure_counter, self.cur_idx, self.reward_sum

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-0.2 * x + 5))

    def is_monotonically_decreasing(self, number_last_el, threshold: float):
        pass
        # for i in range(len(self.reward_sum_list) - 1):
        #     if lst[i] < lst[i + 1]:
        #         return False
        # return True

    def is_monotonically_increasing(self, number_last_el, threshold: float):
        pass

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        # from pathlib import Path
        # import pickle as pkl
        # path_traj = Path.home() / 'TmrlData' / 'reward' / 'traj.pkl'
        # with open(path_traj, 'wb') as file_traj:
        #     pkl.dump(self.traj, file_traj)

        self.cur_idx = 0
        self.prev_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.reward_sum = 0.0
        self.crash_counter = 1

        # self.traj = []
