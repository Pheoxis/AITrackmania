# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging

import wandb

logging.basicConfig(level=logging.INFO)


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """

    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=8,
                 nb_obs_backward=8,
                 nb_zero_rew_before_failure=12,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 max_dist_from_traj=60.0,
                 crash_penalty=10.0,
                 constant_penalty=0.0):
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
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)
        self.survive_reward = 0.5
        self.crash_penalty = crash_penalty
        self.crash_counter = 1
        self.constant_penalty = constant_penalty
        self.gas_reward = 0.0001
        self.brake_penalty = -0.0002
        self.counter = 0
        self.reward_sum = 0.0
        # self.tmp_counter = 0
        # self.traj = []

    def compute_reward(self, pos, crashed: bool = False, gas_input: bool = None, brake_input: bool = None,
                       speed: float = None):
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

        if crashed:
            reward -= abs(self.crash_penalty) * self.crash_counter
            self.crash_counter += 1

        self.reward_sum += reward
        if terminated:
            self.counter += 1
            print(f"Total reward of the run: {self.reward_sum}")
            # logging.info(f"Total reward of the run: {self.reward_sum}")
            # if self.counter % 2 == 0:
            #     self.min_nb_steps_before_failure += 2

        reward -= abs(self.constant_penalty)
        return reward, terminated, self.failure_counter

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
        self.step_counter = 0
        self.failure_counter = 0
        self.reward_sum = 0.0
        self.crash_counter = 1

        # self.traj = []
