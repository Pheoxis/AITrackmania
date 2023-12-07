# standard library imports
import logging
import os
import pickle
import random
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint

logging.basicConfig(level=logging.INFO)

# third-party imports
import numpy as np
# from torch.utils.data import DataLoader, Dataset, Sampler

# local imports
from util import collate_torch
import config.config_constants as cfg
__docformat__ = "google"


def check_samples_crc(original_po, original_a, original_o, original_r, original_d, original_t, rebuilt_po, rebuilt_a,
                      rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t):
    assert original_po is None or str(original_po) == str(
        rebuilt_po), f"previous observations don't match:\noriginal:\n{original_po}\n!= rebuilt:\n{rebuilt_po}"
    assert str(original_a) == str(rebuilt_a), f"actions don't match:\noriginal:\n{original_a}\n!= rebuilt:\n{rebuilt_a}"
    assert str(original_o) == str(
        rebuilt_o), f"observations don't match:\noriginal:\n{original_o}\n!= rebuilt:\n{rebuilt_o}"
    assert str(original_r) == str(rebuilt_r), f"rewards don't match:\noriginal:\n{original_r}\n!= rebuilt:\n{rebuilt_r}"
    assert str(original_d) == str(
        rebuilt_d), f"terminated don't match:\noriginal:\n{original_d}\n!= rebuilt:\n{rebuilt_d}"
    assert str(original_t) == str(
        rebuilt_t), f"truncated don't match:\noriginal:\n{original_t}\n!= rebuilt:\n{rebuilt_t}"
    original_crc = zlib.crc32(str.encode(str((original_a, original_o, original_r, original_d, original_t))))
    crc = zlib.crc32(str.encode(str((rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t))))
    assert crc == original_crc, f"CRC failed: new crc:{crc} != old crc:{original_crc}.\nEither the custom pipeline is corrupted, or crc_debug is False in the rollout worker.\noriginal sample:\n{(original_a, original_o, original_r, original_d)}\n!= rebuilt sample:\n{(rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d)}"
    print("DEBUG: CRC check passed.")


class Memory(ABC):
    """
    Interface implementing the replay buffer.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """

    def __init__(self,
                 device,
                 nb_steps,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
        """
        self.nb_steps = nb_steps
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.sample_preprocessor = sample_preprocessor
        self.crc_debug = crc_debug

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0
        self.average_reward = 0
        self.debug = False

        # init memory
        self.path = Path(dataset_path)
        logging.debug(f"Memory self.path:{self.path}")
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = list(pickle.load(f))
        else:
            logging.info("no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            # self.data = self.data[-self.memory_size:]
            logging.warning(f"the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")
        # random.seed(cfg.SEED)

    def __iter__(self):
        for _ in range(self.nb_steps):
            yield self.sample()

    @abstractmethod
    def append_buffer(self, buffer):
        """
        Must append a Buffer object to the memory.

        Args:
            buffer (tmrl.networking.Buffer): the buffer of samples to append.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Must return the length of the memory.

        Returns:
            int: the maximum `item` argument of `get_transition`

        """
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Must return a transition.

        `info` is required in each sample for CRC debugging. The 'crc' key is what is important when using this feature.

        Args:
            item (int): the index where to sample

        Returns:
            Tuple: (prev_obs, prev_act, rew, obs, terminated, truncated, info)
        """
        raise NotImplementedError

    @abstractmethod
    def collate(self, batch, device):
        """
        Must collate `batch` onto `device`.

        `batch` is a list of training samples.
        The length of `batch` is `batch_size`.
        Each training sample in the list is of the form `(prev_obs, new_act, rew, new_obs, terminated, truncated)`.
        These samples must be collated into 6 tensors of batch dimension `batch_size`.
        These tensors should be collated onto the device indicated by the `device` argument.
        Then, your implementation must return a single tuple containing these 6 tensors.

        Args:
            batch (list): list of `(prev_obs, new_act, rew, new_obs, terminated, truncated)` tuples
            device: device onto which the list needs to be collated into batches `batch_size`

        Returns:
            Tuple of tensors:
            (prev_obs_tens, new_act_tens, rew_tens, new_obs_tens, terminated_tens, truncated_tens)
            collated on device `device`, each of batch dimension `batch_size`
        """
        raise NotImplementedError

    def sample(self):
        indices = self.sample_indices()
        batch = [self[idx] for idx in indices]
        batch = self.collate(batch, self.device)
        return batch

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.append_buffer(buffer)

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, terminated, truncated, info = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d, t = info['crc_sample']
            check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, terminated, truncated = self.sample_preprocessor(
                prev_obs, new_act, rew, new_obs, terminated, truncated
            )
        terminated = np.float32(terminated)  # we don't want bool tensors
        truncated = np.float32(truncated)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, terminated, truncated

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batch_size))


class TorchMemory(Memory, ABC):
    """
    Partial implementation of the `Memory` class collating samples into batched torch tensors.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """

    def __init__(self,
                 device,
                 nb_steps,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
        """
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def collate(self, batch, device):
        return collate_torch(batch, device)


class R2D2Memory(Memory, ABC):
    """
        Partial implementation of the `Memory` class collating samples into batched torch tensors.

        .. note::
           When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
           Your `__init__` method needs to take at least all the arguments of the superclass.
        """

    def __init__(self,
                 device,
                 nb_steps,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False,
                 # info_index=21
                 ):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
        """
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device,
                         # info_index=info_index
                         )
        self.previous_episode = 0
        self.end_episodes_indices = []
        self.chosen_episode = 0
        self.burn_ins = (20, 40)
        self.isNewEpisode = True
        self.chosen_burn_in = 0
        self.reward_sums = []
        self.indices = []
        self.cur_idx = 0
        self.batch_size = batch_size
        self.rewind = cfg.TMRL_CONFIG["ALG"]["R2D2_REWIND"]
        assert 0.1 <= self.rewind <= 0.9, "R2D2 REWIND CONST SHOULD BE BETWEEN 0.1 AND 0.9"

    def collate(self, batch, device):
        return collate_torch(batch, device)

    @staticmethod
    def find_zero_rewards_indices(reward_sums):
        zero_rewards_indices = []
        prev_reward_sum = None

        for i, entry in enumerate(reward_sums):
            reward_sum = entry['reward_sum']
            if prev_reward_sum is not None and reward_sum == 0.0 and prev_reward_sum != 0.0:
                zero_rewards_indices.append(i - 1)

            prev_reward_sum = reward_sum

        return zero_rewards_indices

    @staticmethod
    def normalize_list(input_list):
        # Find the minimum and maximum values in the list
        min_val = min(input_list)
        max_val = max(input_list)

        # Check if the range is zero to avoid division by zero
        if min_val == max_val:
            return [0.0] * len(input_list)

        # Normalize each element in the list
        normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

        return normalized_list

    def sample_indices(self):

        self.end_episodes_indices = self.find_zero_rewards_indices(self.data[19])
        self.reward_sums = [self.data[19][index]['reward_sum'] for index in self.end_episodes_indices]
        # print(self.end_episodes_indices)
        batch_size = self.batch_size

        if len(self.end_episodes_indices) == 0:
            if self.cur_idx == 0:
                self.cur_idx += int(batch_size * self.rewind)
                # self.cur_idx += batch_size // 2
                pom = tuple(range(0, self.cur_idx))
                print(pom)
                return pom
            else:
                if self.cur_idx + batch_size < len(self):
                    pom = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    # self.cur_idx += batch_size // 2
                    self.cur_idx += int(batch_size * self.rewind)
                    print(pom)
                    return pom
                else:
                    pom = tuple(range(len(self) - batch_size, len(self) - 1))
                    print(pom)
                    self.cur_idx = 0
                    return pom
        else:
            if self.isNewEpisode:
                if len(self.reward_sums) == 1:
                    self.chosen_episode = self.end_episodes_indices[0]
                    self.previous_episode = 0
                else:
                    if sum(self.reward_sums) > 0:
                        self.chosen_episode = random.choices(
                            self.end_episodes_indices, weights=self.reward_sums,
                            k=1
                        )[0]
                    else:
                        self.chosen_episode = random.choices(
                            self.end_episodes_indices, weights=self.normalize_list(self.reward_sums),
                            k=1
                        )[0]
                    previous_episode_index = sorted(self.end_episodes_indices).index(self.chosen_episode) - 1
                    if previous_episode_index < 0:
                        self.previous_episode = 0
                    else:
                        self.previous_episode = self.end_episodes_indices[previous_episode_index]

                episode_length = self.chosen_episode - self.previous_episode
                self.chosen_burn_in = random.randint(self.burn_ins[0], self.burn_ins[1])  # Losowanie burn-in

                if episode_length <= batch_size + self.chosen_burn_in:  # kiedy epizod jest krótszy od batch size
                    # print("krótszy batch")
                    result = tuple(range(self.previous_episode, self.chosen_episode - 1))
                    # print(result)
                    return result
                else:
                    # print("pierwszy batch")
                    if self.previous_episode < 0:
                        self.previous_episode = 0

                    self.cur_idx = self.previous_episode + self.chosen_burn_in
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    self.cur_idx += batch_size
                    # print(result)
                    self.isNewEpisode = False
                    return result
            else:
                # self.cur_idx -= batch_size // 2
                self.cur_idx -= int(batch_size * self.rewind)

                if self.cur_idx + batch_size >= self.chosen_episode:  # ostatni batch epizodu
                    # print("ostatni batch")
                    self.isNewEpisode = True

                    result = tuple(range(self.chosen_episode - batch_size, self.chosen_episode - 1))
                    # print(result)
                    self.cur_idx = self.chosen_episode
                    return result
                else:
                    # print("kontynuacja batcha ")
                    self.isNewEpisode = False
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    # print(result)
                    self.cur_idx += batch_size
                    return result

                    self.last_index = self.chosen_episode

        while len(indices) < self.batch_size:
            random_index = random.randint(0, len(self) - 1)
            indices.append(random_index)

        while len(indices) > self.batch_size:
            indices.pop()

        if indices is None:
            raise Exception("Indices cannot be None!")
        # if len(indices) < self.batch_size - 1:
        #     raise Exception("Indices cannot be less!")

        indices = tuple(indices)

        return indices
    # def sample_indices(self):
    #     return tuple(randint(0, len(self) - 1) for _ in range(self.batch_size))


    def sample(self):
        indices = self.sample_indices()
        # print(f"indices[0]: {indices[0]}")
        # print(f"indices[-1]: {indices[-1]}")
        batch = [self[idx] for idx in indices]
        batch = self.collate(batch, self.device)
        return batch


def load_and_print_pickle_file(path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl"):  # r"D:\data2020"
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"nb samples: {len(data[0])}")
    for i, d in enumerate(data):
        print(f"[{i}][0]: {d[0]}")
    print("full data:")
    for i, d in enumerate(data):
        print(f"[{i}]: {d}")


if __name__ == "__main__":
    load_and_print_pickle_file()
