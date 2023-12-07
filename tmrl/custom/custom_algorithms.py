# standard library imports
import itertools
import math
from copy import deepcopy
from dataclasses import dataclass

# third-party imports
import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# local imports
from custom.models import MLPActorCritic, REDQMLPActorCritic
from custom.models.BestActorCriticTQC import QRCNNActorCritic
from custom.utils.nn import copy_shared, no_grad
from util import cached_property
from training import TrainingAgent
import config.config_constants as cfg

import logging

logging.basicConfig(level=logging.INFO)


# Soft Actor-Critic ====================================================================================================


def set_seed(seed=cfg.SEED):
    np.random.seed(seed)
    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)
    # If you're using GPU, set the seed for GPU operations as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set the seed for torch.backends.cudnn if you use cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(eq=False)
class REDQSACAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = REDQMLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning
    learn_entropy_coef: bool = True
    target_entropy: float = None  # if None, the target entropy is set automatically
    n: int = 10  # number of REDQ parallel Q networks
    m: int = 2  # number of REDQ randomly sampled target networks
    q_updates_per_policy_update: int = 1  # in REDQ, this is the "UTD ratio" (20), this interplays with lr_actor

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        set_seed()

        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor, weight_decay=0.001)
        self.q_optimizer_list = [Adam(q.parameters(), lr=self.lr_critic, weight_decay=0.001) for q in self.model.qs]
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)

        self.i_update = 0  # for UTD ratio

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        o, a, r, o2, d, _ = batch

        if update_policy:
            pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        loss_alpha = None
        if self.learn_entropy_coef and update_policy:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (
                    min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        q_prediction_list = [q(o, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)  # * self.n  # averaged for homogeneity with SAC

        for q in self.q_optimizer_list:
            q.zero_grad()
        loss_q.backward()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            qs_pi = [q(o, pi) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            for q in self.model.qs:
                q.requires_grad_(True)

        for q_optimizer in self.q_optimizer_list:
            q_optimizer.step()

        if update_policy:
            self.pi_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            ret_dict = dict(
                loss_actor=self.loss_pi.detach(),
                loss_critic=loss_q.detach(),
            )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# SAC with optional learnable entropy coefficent =======================================================================

@dataclass(eq=False)
class SpinupSacAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        set_seed()
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor, weight_decay=0.005)
        self.q_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                                lr=self.lr_critic, weight_decay=0.005)

        # self.pi_scheduler = CosineAnnealingLR(self.pi_optimizer, T_max=1000, eta_min=self.lr_actor/2)
        # self.q_scheduler = CosineAnnealingLR(self.q_optimizer, T_max=1000, eta_min=self.lr_critic/2)

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

        if cfg.WANDB_GRADIENTS:
            wandb.watch(self.model, log_freq=10)

        # self.is_rendered = False

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        torch.autograd.set_detect_anomaly(True)
        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)
        # if not self.is_rendered or self.is_rendered is None:
        #     torch.onnx.export(self.model.q1, (o, a), "sacv2_model_critic.onnx", verbose=True,
        #                       input_names=["api", "frame"], output_names=["actions"])
        #     torch.onnx.export(self.model.actor, o2, "sacv2_model_actor.onnx", verbose=True,
        #                       input_names=["api", "frame"], output_names=["actions"])
        #     self.is_rendered = True
        # if not self.is_rendered or self.is_rendered is None:
        #     from torchviz import make_dot
        #     make_dot()
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t  # (1, )

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        # Run one gradient descent step for Q1 and Q2

        # loss_q:

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)  # (batch size)
            q2_pi_targ = self.model_target.q2(o2, a2)  # (batch size)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # (batch size)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_critic = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_critic.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Next run one gradient descent step for actor.

        # loss_pi:

        # pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_actor = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()  # actor
        loss_actor.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # FIXME: remove debug info
        with torch.no_grad():

            if not cfg.DEBUG_MODE:
                ret_dict = dict(
                    loss_actor=loss_actor.detach(),
                    loss_critic=loss_critic.detach(),
                )
            else:
                q1_o2_a2 = self.model.q1(o2, a2)
                q2_o2_a2 = self.model.q2(o2, a2)
                q1_targ_pi = self.model_target.q1(o, pi)
                q2_targ_pi = self.model_target.q2(o, pi)
                q1_targ_a = self.model_target.q1(o, a)
                q2_targ_a = self.model_target.q2(o, a)

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()

                ret_dict = dict(
                    loss_actor=loss_actor.detach(),
                    loss_critic=loss_critic.detach(),
                    # debug:
                    debug_log_pi=logp_pi.detach().mean(),
                    debug_log_pi_std=logp_pi.detach().std(),
                    debug_logp_a2=logp_a2.detach().mean(),
                    debug_logp_a2_std=logp_a2.detach().std(),
                    debug_q_a1=q_pi.detach().mean(),
                    debug_q_a1_std=q_pi.detach().std(),
                    debug_q_a1_targ=q_pi_targ.detach().mean(),
                    debug_q_a1_targ_std=q_pi_targ.detach().std(),
                    debug_backup=backup.detach().mean(),
                    debug_backup_std=backup.detach().std(),
                    debug_q1=q1.detach().mean(),
                    debug_q1_std=q1.detach().std(),
                    debug_q2=q2.detach().mean(),
                    debug_q2_std=q2.detach().std(),
                    debug_diff_q1=diff_q1_backup.mean(),
                    debug_diff_q1_std=diff_q1_backup.std(),
                    debug_diff_q2=diff_q2_backup.mean(),
                    debug_diff_q2_std=diff_q2_backup.std(),
                    debug_diff_r_q1=diff_q1_backup_r.mean(),
                    debug_diff_r_q1_std=diff_q1_backup_r.std(),
                    debug_diff_r_q2=diff_q2_backup_r.mean(),
                    debug_diff_r_q2_std=diff_q2_backup_r.std(),
                    debug_diff_q1pt_qpt=diff_q1pt_qpt.mean(),
                    debug_diff_q2pt_qpt=diff_q2pt_qpt.mean(),
                    debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean(),
                    debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean(),
                    debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean(),
                    debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean(),
                    debug_diff_q1_q1t_a=diff_q1_q1t_a.mean(),
                    debug_diff_q2_q2t_a=diff_q2_q2t_a.mean(),
                    debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std(),
                    debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std(),
                    debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std(),
                    debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std(),
                    debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std(),
                    debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std(),
                    debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std(),
                    debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std(),
                    debug_r=r.detach().mean(),
                    debug_r_std=r.detach().std(),
                    debug_d=d.detach().mean(),
                    debug_d_std=d.detach().std(),
                    debug_a_0=a[:, 0].detach().mean(),
                    debug_a_0_std=a[:, 0].detach().std(),
                    debug_a_1=a[:, 1].detach().mean(),
                    debug_a_1_std=a[:, 1].detach().std(),
                    debug_a_2=a[:, 2].detach().mean(),
                    debug_a_2_std=a[:, 2].detach().std(),
                    debug_a1_0=pi[:, 0].detach().mean(),
                    debug_a1_0_std=pi[:, 0].detach().std(),
                    debug_a1_1=pi[:, 1].detach().mean(),
                    debug_a1_1_std=pi[:, 1].detach().std(),
                    debug_a1_2=pi[:, 2].detach().mean(),
                    debug_a1_2_std=pi[:, 2].detach().std(),
                    debug_a2_0=a2[:, 0].detach().mean(),
                    debug_a2_0_std=a2[:, 0].detach().std(),
                    debug_a2_1=a2[:, 1].detach().mean(),
                    debug_a2_1_std=a2[:, 1].detach().std(),
                    debug_a2_2=a2[:, 2].detach().mean(),
                    debug_a2_2_std=a2[:, 2].detach().std(),
                )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict

    # TQC with learnable entropy coefficent ===================================================================


# https://github.com/yosider/ml-agents-1/blob/master/docs/Training-SAC.md
@dataclass(eq=False)
class TQCAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None
    model_cls: type = QRCNNActorCritic  # Replace with your QuantileActorCritic class
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None
    top_quantiles_to_drop: int = 3  # ~8% of total number of atoms
    quantiles_number: int = 32

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)  # Use TQCActorCritic
        logging.debug(f" device TQC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor, weight_decay=0.005)
        self.critic_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                                     lr=self.lr_critic, weight_decay=0.005)

        if len(cfg.SCHEDULER_CONFIG["NAME"]) > 0:
            self.actor_scheduler = CosineAnnealingWarmRestarts(
                self.actor_optimizer,
                cfg.SCHEDULER_CONFIG["T_0"],
                cfg.SCHEDULER_CONFIG["T_mult"],
                cfg.SCHEDULER_CONFIG["eta_min"],
                cfg.SCHEDULER_CONFIG["last_epoch"]
            )

            self.critic_scheduler = CosineAnnealingWarmRestarts(
                self.critic_optimizer,
                cfg.SCHEDULER_CONFIG["T_0"],
                cfg.SCHEDULER_CONFIG["T_mult"],
                cfg.SCHEDULER_CONFIG["eta_min"],
                cfg.SCHEDULER_CONFIG["last_epoch"]
            )

        self.quantiles_total = self.model.q1.num_quantiles + self.model.q2.num_quantiles

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(device)

        if cfg.WANDB_GRADIENTS:
            wandb.watch(self.model, log_freq=10)

    def get_actor(self):
        return self.model_nograd.actor

    @staticmethod
    def calculate_huber_loss(td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        return loss

    @staticmethod
    def clip_weights(model, max_value=0.98):
        for param in model.parameters():
            param.data.clamp_(-max_value, max_value)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.polyak * local_param.data + (1.0 - self.polyak) * target_param.data)

    # https://github.com/SamsungLabs/tqc_pytorch/blob/master/tqc/trainer.py
    def quantile_huber_loss_f(self, quantiles, samples):
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        huber_loss = self.calculate_huber_loss(pairwise_delta)

        n_quantiles = quantiles.shape[2]
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

    @staticmethod
    def clip_weights(model, max_value=0.98):
        for param in model.parameters():
            param.data.clamp_(-max_value, max_value)

    def train(self, epoch, batch, batch_index, iters):
        o, a, r, o2, d, _ = batch

        batch_size = r.shape[0]
        pi, logp_pi = self.model.actor(o)

        # loss_alpha:

        alpha_loss = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)

        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/tqc/tqc.py :
        with torch.no_grad():
            new_next_action, next_log_pi = self.model.actor(o2)  # Tensor(batch x 3), Tensor(batch)
            # Compute and cut quantiles at the next state
            next_q1 = self.model_target.q1(o2, new_next_action)  # Tensor(batch x quantiles)
            next_q2 = self.model_target.q2(o2, new_next_action)  # Tensor(batch x quantiles)
            next_z = torch.stack((next_q1, next_q2), dim=1)  # Tensor(batch x nets x quantiles)
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))  # Tensor(batch x [nets * quantiles])
            sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
            target_quantiles = sorted_z_part - alpha_t * next_log_pi.reshape(-1, 1)
            not_done = 1 - d
            tmp = target_quantiles * not_done[:, None]
            r = r.unsqueeze(1)
            r = r.expand(-1, self.quantiles_total - self.top_quantiles_to_drop)
            backup = r + self.gamma * tmp

        cur_z = torch.stack((q1_pi, q2_pi), dim=1)
        critic_loss = self.quantile_huber_loss_f(cur_z, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step(epoch + batch_index / iters)  # TODO: idk czy dobrze

        new_action, log_pi = self.get_actor()(o)
        q1_pi = self.model.q1(o, new_action)
        q2_pi = self.model.q2(o, new_action)
        new_critic = torch.stack((q1_pi, q2_pi), dim=1)
        actor_loss = (alpha_t * log_pi - new_critic.mean(2).mean(1, keepdim=True)).mean()

        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # self.clip_weights(self.model.q1)
        # self.clip_weights(self.model.q2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step(epoch + batch_index / iters)  # TODO: idk czy dobrze
        # self.clip_weights(self.model.actor)

        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


        ret_dict = {
            "loss_actor": actor_loss.detach().item(),
            "loss_critic": critic_loss.detach().item(),  # or any other relevant critic loss
            "actor_lr": self.actor_optimizer.param_groups[0]['lr'],
            "critic_lr": self.critic_optimizer.param_groups[0]['lr']
        }

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = alpha_loss.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# =================================== IQN ================================================================

@dataclass(eq=False)
class IQNAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None
    model_cls: type = QRCNNActorCritic  # Replace with your QuantileActorCritic class
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None
    quantiles_number: int = 32
    quantile_embedding_size: int = 128

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        self.pis = torch.FloatTensor(
            [np.pi * i for i in range(self.quantile_embedding_size)]
        ).view(1, 1, self.quantile_embedding_size).to(self.device)

        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device IQN: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor, weight_decay=0.005)
        self.critic_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                                     lr=self.lr_critic, weight_decay=0.005)

        self.quantiles_total = self.model.q1.num_quantiles + self.model.q2.num_quantiles

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(device)

        if cfg.WANDB_GRADIENTS:
            wandb.watch(self.model, log_freq=10)

    def get_actor(self):
        return self.model_nograd.actor

    @staticmethod
    def calculate_huber_loss(td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        # assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
        return loss

    @staticmethod
    def clip_weights(model, max_value=0.98):
        for param in model.parameters():
            param.data.clamp_(-max_value, max_value)

    def calc_cos(self, batch_size, n_tau=quantile_embedding_size):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1)  # (batch_size, n_tau, 1)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.quantile_embedding_size), "cos shape is incorrect"
        return cos, taus

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.polyak * local_param.data + (1.0 - self.polyak) * target_param.data)

    def train(self, batch):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        torch.autograd.set_detect_anomaly(True)
        o, a, r, o2, d, _ = batch
        batch_size = len(batch[0])
        pi, logp_pi = self.model.actor(o)

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t  # (1, )

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target(o2)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)  # (batch_size, 1, N)

        # Compute Q targets for current states
        Q_targets = r.unsqueeze(-1) + (self.gamma ** self.n_step * Q_targets_next * (1. - d.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(o)
        Q_expected = Q_expected.gather(2, a.unsqueeze(-1).expand(batch_size, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (batch_size, 8, 8), "wrong td error shape"
        huber_l = self.calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1)  # , keepdim=True if per weights get multipl
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)


        #     ret_dict = {
        #         "loss_actor": actor_loss.item(),
        #         "loss_critic": critic_loss.item(),  # or any other relevant critic loss
        #      }
        #
        #     if self.learn_entropy_coef:
        #         ret_dict["loss_entropy_coef"] = alpha_loss.item()
        #         ret_dict["entropy_coef"] = alpha_t.item()
        #
        # return ret_dict

