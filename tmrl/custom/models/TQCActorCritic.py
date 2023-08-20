# Import necessary libraries
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from torch.distributions import Normal
from MobileNetActorCritic import  SquashedActorMobileNetV3

# ... (other imports and definitions remain the same)
def rnn(input_size, rnn_size, rnn_len):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=0, bidirectional=False
    )
    return gru
def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

class MobileNetQuantileQFunction(nn.Module):
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU, num_quantiles=64):
        super().__init__()
        dim_obs = sum(np.prod(space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [num_quantiles], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.num_quantiles = num_quantiles

    def forward(self, obs_seq, act, save_hidden=False):
        self.rnn.flatten_parameters()
        batch_size = obs_seq[0].shape[0]
        tmp = obs_seq[3][0][0]
        x = tmp.permute(0, 3, 1, 2)
        output = self.mobilenet(x)
        dupa = list(obs_seq)
        dupa[3] = output.unsqueeze(0)

        if not save_hidden or self.h is None:
            device = dupa[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        max_dim2_size = max(o.shape[2] for o in dupa)
        max_dim3_size = max(o.shape[3] for o in dupa)
        obs_seq_resized = [
            F.adaptive_avg_pool2d(o, output_size=(max_dim2_size, max_dim3_size)) for o in dupa
        ]

        obs_seq_resized = [o.squeeze(dim=3).permute(0, 1, 3, 2) for o in obs_seq_resized]

        obs_seq_cat = torch.cat(obs_seq_resized, dim=1)

        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = torch.cat((net_out, act), -1)
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        q = self.mlp(net_out)

        return q

class MobileNetQuantileActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, rnn_size=100,
                 rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU, num_quantiles=64, q_learning_rate=1e-3):
        super().__init__()

        self.actor = SquashedActorMobileNetV3(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q1 = MobileNetQuantileQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation, num_quantiles)
        self.q2 = MobileNetQuantileQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation, num_quantiles)
        self.num_quantiles = num_quantiles

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=q_learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=q_learning_rate)

    def compute_quantile_targets(self, obs_seq, actions, rewards, next_obs_seq, dones, discount_factor):
        with torch.no_grad():
            next_actions, _ = self.actor.forward(next_obs_seq)
            q1_targets_next = self.q1.forward(next_obs_seq, next_actions)
            q2_targets_next = self.q2.forward(next_obs_seq, next_actions)
            q_targets_next = torch.min(q1_targets_next, q2_targets_next)
            quantile_rewards = rewards.unsqueeze(1) + discount_factor * (1 - dones.unsqueeze(1)) * q_targets_next
        return quantile_rewards

    def huber_loss(self, errors, kappa=1.0):
        return torch.where(torch.abs(errors) <= kappa, 0.5 * errors ** 2, kappa * (torch.abs(errors) - 0.5 * kappa))

    def update_quantile_q_functions(self, obs_seq, actions, quantile_targets, kappa=1.0):
        q1_predictions = self.q1.forward(obs_seq, actions)
        q2_predictions = self.q2.forward(obs_seq, actions)

        quantile_errors1 = quantile_targets.unsqueeze(2) - q1_predictions.unsqueeze(1)
        quantile_errors2 = quantile_targets.unsqueeze(2) - q2_predictions.unsqueeze(1)

        huber_loss1 = self.huber_loss(quantile_errors1, kappa)
        huber_loss2 = self.huber_loss(quantile_errors2, kappa)

        q1_loss = huber_loss1.mean(dim=2).sum(dim=1).mean()
        q2_loss = huber_loss2.mean(dim=2).sum(dim=1).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        return q1_loss.item(), q2_loss.item()