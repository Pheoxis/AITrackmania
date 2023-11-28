import datetime
import json
import logging
import math
import threading
from argparse import ArgumentParser

import keyboard
from inputs import get_gamepad

import config.config_constants as cfg
from custom.interfaces.TM2020InterfaceTQCmini import TM2020InterfaceTQCmini
from networking import HumanWorker
from tmrl import GenericGymEnv
from util import partial
import config.config_objects as cfg_obj


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):
        self.LeftJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0

        self._monitor_thread = threading.Thread(target=self.get_actions, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def get_actions(self):
        events = get_gamepad()
        for event in events:
            if event.code == 'ABS_X':
                self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL  # normalize between -1 and 1
                # print(self.LeftJoystickX)
            elif event.code == 'ABS_Z':
                self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL  # normalize between 0 and 1
                # print(self.LeftTrigger)
            elif event.code == 'ABS_RZ':
                self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL  # normalize between 0 and 1
                # print(self.RightTrigger)

        return self.RightTrigger, self.LeftTrigger, self.LeftJoystickX


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


"""
1. Press gas pedal to start recording
2. Start record
3. if terminated stop recording 
"""
"""
Send inputs to trainer
"""


def record_episode():
    controller = XboxController()
    config = cfg_obj.CONFIG_DICT

    hw = HumanWorker(env_cls=partial(GenericGymEnv, id="real-time-gym-v1", gym_kwargs={"config": config}),
                     device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
                     server_ip=cfg.SERVER_IP_FOR_WORKER,
                     standalone=False)

    is_recording = False

    while True:
        actions = controller.get_actions()
        if actions[0] > 0.5 and not is_recording:  # change to if xbox controller right trigger > 0.5
            logging.info(f"start recording")
            is_recording = True
            print(f"is_recording: {is_recording}")
        if is_recording:
            hw.collect_train_episode(controller)
            if keyboard.is_pressed('q'):
                break



    # rw = RolloutWorker(env_cls=partial(GenericGymEnv, id="real-time-gym-v1", gym_kwargs={"config": config}),
    #                    actor_module_cls=cfg_obj.POLICY,
    #                    sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
    #                    device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
    #                    server_ip=cfg.SERVER_IP_FOR_WORKER,
    #                    max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
    #                    model_path=cfg.MODEL_PATH_WORKER,
    #                    obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
    #                    crc_debug=cfg.CRC_DEBUG,
    #                    standalone=False)

    # stop_recording = False
    # is_recording = False
    # obs = None
    # info = None
    #
    # buffer = Buffer()
    # while not stop_recording:
    #     terminated = False
    #     sample_size = 0
    #     ret = 0.0
    #     while not terminated:
    #         actions = controller.get_actions()
    #         if actions[0] > 0.5 and not is_recording:  # change to if xbox controller right trigger > 0.5
    #             logging.info(f"start recording")
    #             is_recording = True
    #             obs, info = rw.reset(collect_samples=True)
    #             print(f"is_recording: {is_recording}")
    #         if is_recording:
    #             obs, rew, terminated, truncated, info = rw.step(obs=obs, test=False, collect_samples=True, last_step=False)
    #             ret += rew
    #             sample_size += 1
    #             if keyboard.is_pressed('q'):
    #                 terminated = True
    #             print(f"terminated: {terminated}")
    #         print(f"actions: {actions}")
    #         print(f"obs: {obs}")
    #         print(f"info: {info}")
    #     buffer.stat_train_steps = sample_size
    #     buffer.stat_train_return = ret
    #     rw.send_and_clear_buffer()


"""
episode = 0
        while episode < nb_episodes:
            if episode % test_episode_interval == 0 and not self.crc_debug:
                print_with_timestamp("running test episode")
                self.run_episode(self.max_samples_per_episode, train=False)
            print_with_timestamp("collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            print_with_timestamp("copying buffer for sending")
            self.send_and_clear_buffer()
            print_with_timestamp("checking for new weights")
            self.update_actor_weights()
            episode += 1
"""


# def step(obs, act, collect_samples, rw, last_step=False):
#     new_obs, rew, terminated, truncated, info = rw.env.unwrapped.step_without_act(action=act)
#
#     if rw.obs_preprocessor is not None:
#         new_obs = rw.obs_preprocessor(new_obs)
#     if collect_samples:
#         if last_step and not terminated:
#             truncated = True
#         if rw.crc_debug:
#             info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
#         if rw.get_local_buffer_sample:
#             sample = rw.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
#         else:
#             sample = act, new_obs, rew, terminated, truncated, info
#         rw.buffer.append_sample(
#             sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
#     return new_obs, rew, terminated, truncated, info


if __name__ == '__main__':
    record_episode()
