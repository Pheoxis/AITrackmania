# standard library imports
import itertools
<<<<<<< HEAD
import logging
=======
import math
>>>>>>> origin/main
from copy import deepcopy
from dataclasses import dataclass

# third-party imports
import numpy as np
import torch
from torch.optim import Adam

import config.config_constants as cfg
import wandb
# local imports
from custom.models import MLPActorCritic, REDQMLPActorCritic
<<<<<<< HEAD
from custom.models.TQCActorCritic import QuantileActorCritic
=======
from custom.models.BestActorCriticTQC import QRCNNActorCritic
>>>>>>> origin/main
from custom.utils.nn import copy_shared, no_grad
from training import TrainingAgent
from util import cached_property

<<<<<<< HEAD
=======
import logging

>>>>>>> origin/main
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

        # self.pi_scheduler = ReduceLROnPlateau(self.pi_optimizer, mode='max', patience=4, cooldown=1, eps=1)
        # self.q_optimizer = ReduceLROnPlateau(self.q_optimizer, 'min', cooldown=1)

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

        self.pi_optimizer.zero_grad()
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

<<<<<<< HEAD
=======
    # SAC with optional learnable entropy coefficent ===================================================================


>>>>>>> origin/main
@dataclass(eq=False)
class TQCAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None
<<<<<<< HEAD
    model_cls: type = QuantileActorCritic  # Replace with your QuantileActorCritic class
=======
    model_cls: type = QRCNNActorCritic  # Replace with your QuantileActorCritic class
>>>>>>> origin/main
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None
<<<<<<< HEAD
    top_quantiles_to_drop: int = 5
=======
    top_quantiles_to_drop: int = 2  # ~8% of total number of atoms
    quantiles_number: int = 25
>>>>>>> origin/main

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
<<<<<<< HEAD
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)  # Use TQCActorCritic
=======
        quantiles_number = self.quantiles_number
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space, num_quantiles=quantiles_number)  # Use TQCActorCritic
>>>>>>> origin/main
        logging.debug(f" device TQC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
<<<<<<< HEAD
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                                lr=self.lr_critic)
=======
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor, weight_decay=0.001)
        self.critic_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                                     lr=self.lr_critic, weight_decay=0.001)

        self.quantiles_total = self.model.q1.num_quantiles + self.model.q2.num_quantiles
>>>>>>> origin/main

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(device)

<<<<<<< HEAD
    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        o, a, r, o2, d, _ = batch

=======
        if cfg.WANDB_GRADIENTS:
            wandb.watch(self.model, log_freq=10)

    def get_actor(self):
        return self.model_nograd.actor

    # @staticmethod
    # def huber_loss(errors, kappa=1.0):
    #     return torch.where(torch.abs(errors) <= kappa, 0.5 * errors ** 2, kappa * (torch.abs(errors) - 0.5 * kappa))
    # # https://github.com/SamsungLabs/tqc_pytorch/blob/master/tqc/trainer.py

    def quantile_huber_loss_f(self, quantiles, samples):
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)

        n_quantiles = quantiles.shape[2]
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

    def train(self, batch):
        o, a, r, o2, d, _ = batch

        batch_size = r.shape[0]
>>>>>>> origin/main
        pi, logp_pi = self.model.actor(o)

        # loss_alpha:

<<<<<<< HEAD
        loss_alpha = None
=======
        alpha_loss = None
>>>>>>> origin/main
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
<<<<<<< HEAD
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
=======
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
>>>>>>> origin/main
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
<<<<<<< HEAD
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()


        quantile_targets = self.compute_quantile_targets(o2, a, r, o2, d, self.gamma)
        q1_loss, q2_loss = self.update_quantile_q_functions(o, a, quantile_targets)

=======
        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

>>>>>>> origin/main
        alpha = torch.exp(self.log_alpha)

        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
<<<<<<< HEAD
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        with torch.no_grad():
            new_next_action, next_log_pi = self.get_actor()(o2)
            # Compute and cut quantiles at the next state
            next_z = self.model_target.critic(o2, new_next_action)
            sorted_z, _ = torch.sort(next_z.reshape(batch, -1))
            sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
            # Compute target
            target = r + d * self.gamma * (sorted_z_part - alpha * next_log_pi)

        cur_z = self.model.critic(o, a)
        critic_loss = self.quantile_huber_loss_f(cur_z, target)

        new_action, log_pi = self.get_actor()(o)
        #alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.model.critic(o, new_action).mean(2).mean(1, keepdim=True)).mean()

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        for param, target_param in zip(self.model.critic.parameters(), self.model_target.critic.parameters()):
            target_param.data.copy_(self.polyak * param.data + (1 - self.polyak) * target_param.data)

        self.pi_optimizer.zero_grad()
        actor_loss.backward()
        self.pi_optimizer.step()

        ret_dict = {
            "loss_actor": loss_pi.detach(),
            "loss_critic": (q1_loss + q2_loss) / 2,  # or any other relevant critic loss
            "loss_alpha": loss_alpha.detach() if self.learn_entropy_coef else None,
            "alpha": alpha_t.item(),
        }

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


=======

        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/tqc/tqc.py :
        with torch.no_grad():
            new_next_action, next_log_pi = self.model.actor(o2)  # Tensor(batch x 3), Tensor(batch)
            # Compute and cut quantiles at the next state
            next_q1 = self.model_target.q1(o2, new_next_action)  # Tensor(batch x quantiles)
            next_q2 = self.model_target.q2(o2, new_next_action)  # Tensor(batch x quantiles)
            next_z = torch.stack((next_q1, next_q2), dim=1)  # Tensor(batch x nets x quantiles)
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))  # Tensor(batch x [nets * quantiles])
            sorted_z_part = sorted_z[:,
                            :self.quantiles_total - self.top_quantiles_to_drop]
            target_quantiles = sorted_z_part - alpha * next_log_pi.reshape(-1, 1)
            not_done = 1 - d
            tmp = target_quantiles * not_done[:, None]
            r = r.unsqueeze(1)
            r = r.expand(-1, self.quantiles_total - self.top_quantiles_to_drop)
            backup = r + tmp

        cur_z = torch.stack((q1_pi, q2_pi), dim=1)
        critic_loss = self.quantile_huber_loss_f(cur_z, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        new_action, log_pi = self.get_actor()(o)
        q1_pi = self.model.q1(o, new_action)
        q2_pi = self.model.q2(o, new_action)
        new_critic = torch.stack((q1_pi, q2_pi), dim=1)
        actor_loss = (alpha * log_pi - new_critic.mean(2).mean(1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        ret_dict = {
            "loss_actor": actor_loss.detach(),
            "loss_critic": critic_loss.detach(),  # or any other relevant critic loss
        }

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = alpha_loss.detach()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict
>>>>>>> origin/main
