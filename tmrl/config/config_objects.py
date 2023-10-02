# third-party imports
# from tmrl.custom.custom_checkpoints import load_run_instance_images_dataset, dump_run_instance_images_dataset
# third-party imports
import rtgym

# local imports
import config.config_constants as cfg
import custom.models.BestActorCriticTQC as tqc
import custom.models.MaybeBetterTQC as mtqc
import custom.models.BetterTQCmini as mini
from custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from custom.custom_algorithms import TQCAgent as TQC_Agent
from custom.custom_checkpoints import update_run_instance
from custom.custom_memories import MemoryTMLidar, MemoryTMLidarProgress, get_local_buffer_sample_lidar, \
    get_local_buffer_sample_lidar_progress, get_local_buffer_sample_tm20_imgs, MemoryTMBest, \
    get_local_buffer_sample_mobilenet, MemoryTMFull, MemoryR2D2, MemoryR2D2mini
from custom.custom_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, \
    obs_preprocessor_tm_lidar_progress_act_in_obs, obs_preprocessor_mobilenet_act_in_obs

from custom.interfaces.TM2020InterfaceTQCmini import TM2020InterfaceTQCmini
# from custom.interfaces.TM2020InterfaceTQC import TM2020InterfaceTQC
from custom.interfaces.TM2020Interface import TM2020Interface
from custom.interfaces.TM2020InterfaceCustom import TM2020InterfaceCustom
from custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress
from custom.interfaces.TM2020InterfaceTrackMap import TM2020InterfaceTrackMap
from custom.models.BestActorCritic import RCNNActorCritic, SquashedActorRCNN
from custom.models.MLPActorCritic import MLPActorCritic, SquashedGaussianMLPActor
from custom.models.MobileNetActorCritic import MobileNetActorCritic, SquashedActorMobileNetV3
from custom.models.REDQMLPActorCritic import REDQMLPActorCritic
from custom.models.RNNActorCritic import RNNActorCritic, SquashedGaussianRNNActor
from custom.models.VanillaCNNActorCritic import VanillaCNNActorCritic, SquashedGaussianVanillaCNNActor
from custom.models.VanillaColorCNNActorCritic import VanillaColorCNNActorCritic, SquashedGaussianVanillaColorCNNActor
from envs import GenericGymEnv
from training_offline import TorchTrainingOffline
from util import partial

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]
assert ALG_NAME in ["SAC", "REDQSAC",
                    "TQC"], f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."

# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    else:
        TRAIN_MODEL = MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic
        POLICY = SquashedGaussianMLPActor
else:
    if cfg.PRAGMA_CUSTOM:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = MobileNetActorCritic
        POLICY = SquashedActorMobileNetV3
    elif cfg.PRAGMA_BEST:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RCNNActorCritic
        POLICY = SquashedActorRCNN
    elif cfg.PRAGMA_BEST_TQC:
        assert ALG_NAME == "TQC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = tqc.QRCNNActorCritic
        POLICY = tqc.SquashedActorQRCNN
    elif cfg.PRAGMA_MBEST_TQC:
        # assert ALG_NAME == "TQC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = mini.QRCNNActorCritic
        POLICY = mini.SquashedActorQRCNN
        # TRAIN_MODEL = mtqc.QRCNNActorCritic
        # POLICY = mtqc.SquashedActorQRCNN
    else:
        assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = VanillaCNNActorCritic if cfg.GRAYSCALE else VanillaColorCNNActorCritic
        POLICY = SquashedGaussianVanillaCNNActor if cfg.GRAYSCALE else SquashedGaussianVanillaColorCNNActor

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        INT = partial(TM2020InterfaceLidarProgress, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
    elif cfg.PRAGMA_TRACKMAP:
        INT = partial(TM2020InterfaceTrackMap, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
    else:
        INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        INT = partial(
            TM2020InterfaceTQCmini, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE, resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
            crash_penalty=cfg.CRASH_PENALTY, constant_penalty=cfg.CONSTANT_PENALTY,
            checkpoint_reward=cfg.CHECKPOINT_REWARD, lap_reward=cfg.LAP_REWARD,
            min_nb_steps_before_failure=200 if cfg.MAP_NAME == "tmrl_test" else 120
        )
        # INT = partial(
        #     TM2020InterfaceTQC, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD,
        #     grayscale=cfg.GRAYSCALE, resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        #     crash_penalty=cfg.CRASH_PENALTY, constant_penalty=cfg.CONSTANT_PENALTY,
        #     checkpoint_reward=cfg.CHECKPOINT_REWARD, lap_reward=cfg.LAP_REWARD,
        #     min_nb_steps_before_failure=200 if cfg.MAP_NAME == "tmrl_test" else 120
        # )
        # INT = partial(
        #     TM2020InterfaceCustom, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD,
        #     grayscale=cfg.GRAYSCALE, resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        #     crash_penalty=cfg.CRASH_PENALTY, constant_penalty=cfg.CONSTANT_PENALTY,
        #     min_nb_steps_before_failure=200 if cfg.MAP_NAME == "tmrl_test" else 120
        # )
    else:
        INT = partial(TM2020Interface,
                      img_hist_len=cfg.IMG_HIST_LEN,
                      gamepad=cfg.PRAGMA_GAMEPAD,
                      grayscale=cfg.GRAYSCALE,
                      resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# to compress a sample before sending it over the local network/Internet:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_mobilenet
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs

# to preprocess observations that come out of the gymnasium environment:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC:
        OBS_PREPROCESSOR = obs_preprocessor_mobilenet_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer:
SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert False, "not implemented"
    else:
        if cfg.PRAGMA_PROGRESS:
            MEM = MemoryTMLidarProgress
        else:
            MEM = MemoryTMLidar
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC:
        MEM = MemoryTMBest
    elif cfg.PRAGMA_MBEST_TQC:
        # MEM = MemoryR2D2
        MEM = MemoryR2D2mini
    else:
        MEM = MemoryTMFull

MEMORY = partial(MEM,
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 dataset_path=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG)

# ALGORITHM: ===================================================

if ALG_NAME == "SAC":
    AGENT = partial(
        SAC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"]  # inverse of reward scale
    )
elif ALG_NAME == "TQC":
    AGENT = partial(
        TQC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        top_quantiles_to_drop=ALG_CONFIG["TOP_QUANTILES_TO_DROP"],
        quantiles_number=ALG_CONFIG["QUANTILES_NUMBER"]
    )
else:
    AGENT = partial(
        REDQ_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        n=ALG_CONFIG["REDQ_N"],  # number of Q networks
        m=ALG_CONFIG["REDQ_M"],  # number of Q targets
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
    )


# TRAINER: =====================================================


def sac_v2_entropy_scheduler(agent, epoch):
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch


ENV_CLS = partial(GenericGymEnv, id="real-time-gym-v1", gym_kwargs={"config": CONFIG_DICT})

if cfg.PRAGMA_LIDAR:  # lidar
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  # set this > 0 to start from an existing
    # policy (fills the buffer up to this number of samples before starting training)
else:  # images
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])

# CHECKPOINTS: ===================================================

DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # dump_run_instance_images_dataset
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # load_run_instance_images_dataset
UPDATER_FN = update_run_instance if ALG_NAME in ["SAC", "REDQSAC", "TQC"] else None
