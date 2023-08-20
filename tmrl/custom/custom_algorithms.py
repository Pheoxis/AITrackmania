# standard library imports
import itertools
from copy import deepcopy
from dataclasses import dataclass
from torch.nn import Module, Linear
from torch.nn.functional import relu, logsigmoid
from torch.distributions import Distribution, Normal
import torch.optim as optim
from models.RNNActorCritic import SquashedGaussianRNNActor

# third-party imports
import numpy as np
import torch
from torch.optim import Adam

# local imports
from custom.models import MLPActorCritic, REDQMLPActorCritic
from custom.utils.nn import copy_shared, no_grad
from util import cached_property
from training import TrainingAgent
import config.config_constants as cfg

import logging
logging.basicConfig(level=logging.INFO)

# Soft Actor-Critic ====================================================================================================


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
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer_list = [Adam(q.parameters(), lr=self.lr_critic) for q in self.model.qs]
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
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

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
            self.loss_pi = loss_pi.detach()
        ret_dict = dict(
            loss_actor=self.loss_pi,
            loss_critic=loss_q.detach(),
        )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# REDQ-SAC =============================================================================================================

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
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), lr=self.lr_critic)

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

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)
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
            alpha_t = self.alpha_t

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
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_q.backward()
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
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
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
                    loss_actor=loss_pi.detach(),
                    loss_critic=loss_q.detach(),
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
                    loss_actor=loss_pi.detach(),
                    loss_critic=loss_q.detach(),
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

@dataclass(eq=False)
class TQCAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None
    model_cls: type = MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function
        self.pi_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = optim.Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                                      lr=self.lr_critic)

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch)
        alpha = torch.exp(self.log_alpha)

        with torch.no_grad():
            new_next_action, next_log_pi = self.get_actor()(next_state)
            # Compute and cut quantiles at the next state
            next_z = self.model_target.critic(next_state, new_next_action)
            sorted_z, _ = torch.sort(next_z.reshape(batch, -1))
            sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]
            # Compute target
            target = reward + not_done * self.gamma * (sorted_z_part - alpha * next_log_pi)

        cur_z = self.model.critic(state, action)
        critic_loss = self.quantile_huber_loss_f(cur_z, target)

        new_action, log_pi = self.get_actor()(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.model.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        for param, target_param in zip(self.model.critic.parameters(), self.model_target.critic.parameters()):
            target_param.data.copy_(self.polyak * param.data + (1 - self.polyak) * target_param.data)

        self.pi_optimizer.zero_grad()
        actor_loss.backward()
        self.pi_optimizer.step()

        if self.learn_entropy_coef:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.total_it += 1
    def quantile_huber_loss_f(self,quantiles, samples):
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1,
                                 abs_pairwise_delta - 0.5,
                                 pairwise_delta ** 2 * 0.5)

        n_quantiles = quantiles.shape[2]
        tau = torch.arange(n_quantiles, device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss




















    class Mlp(Module):
        def __init__(
                self,
                input_size,
                hidden_sizes,
                output_size
        ):
            super().__init__()
            # TODO: initialization
            self.fcs = []
            in_size = input_size
            for i, next_size in enumerate(hidden_sizes):
                fc = Linear(in_size, next_size)
                self.add_module(f'fc{i}', fc)
                self.fcs.append(fc)
                in_size = next_size
            self.last_fc = Linear(in_size, output_size)

        def forward(self, input):
            h = input
            for fc in self.fcs:
                h = relu(fc(h))
            output = self.last_fc(h)
            return output

    class TanhNormal(Distribution):
        def __init__(self, normal_mean, normal_std):
            super().__init__()
            self.normal_mean = normal_mean
            self.normal_std = normal_std
            self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=self.device or ("cuda" if torch.cuda.is_available() else "cpu")),
                                          torch.ones_like(self.normal_std, device=self.device or ("cuda" if torch.cuda.is_available() else "cpu")))
            self.normal = Normal(normal_mean, normal_std)

        def log_prob(self, pre_tanh):
            log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
            result = self.normal.log_prob(pre_tanh) - log_det
            return result

        def rsample(self):
            pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
            return torch.tanh(pretanh), pretanh