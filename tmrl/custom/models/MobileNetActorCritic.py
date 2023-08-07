import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from torch.distributions import Normal
import torch
from util import prod
from custom.models.MobileNetV3 import mobilenetv3_large
from custom.models.model_constants import LOG_STD_MIN, LOG_STD_MAX
from custom.models.model_blocks import mlp
from actor import TorchActorModule

LOG_STD_MIN = -20
LOG_STD_MAX = 2
# Assuming you have already defined `rnn` and other utility functions
def conv1x1(self, x):
    x = torch.squeeze(x, dim=2)  # Squeeze the third dimension
    x = torch.squeeze(x, dim=2)  # Squeeze the fourth dimension
    return F.conv2d(x, self.conv1x1_weights)

def rnn(input_size, rnn_size, rnn_len):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=0, bidirectional=False
    )
    return gru


# def mlp(sizes, activation=nn.ReLU, output_activation=None):
#     layers = []
#     for i in range(len(sizes) - 1):
#         act = activation if i < len(sizes) - 2 else output_activation
#         layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
#     return nn.Sequential(*layers)


class SquashedActorMobileNetV3(TorchActorModule):
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        self.mobilenet = mobilenetv3_large()
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        # tm = observation_space.spaces
        dim_obs=dim_obs-prod(observation_space.spaces[3].shape)+self.mobilenet.num_classes
        dim_act = action_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act-3] + list(mlp_sizes) + [100], activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.squash_correction = 2 * (np.log(2) - np.log(self.act_limit))
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.conv1x1_weights = nn.Parameter(torch.randn(rnn_size, 1, 1))



    def conv1x1(self, x):
        return F.conv1d(x, self.conv1x1_weights)

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        self.rnn.flatten_parameters()
        batch_size = obs_seq[0].shape[0]
        tmp = obs_seq[3][0][0]
        x = tmp.permute(0, 3, 1, 2)
        output = self.mobilenet(x)
        observation = list(obs_seq)
        observation[3] = output.unsqueeze(0)


        if not save_hidden or self.h is None:
            device = observation[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        # Reshape tensors in obs_seq to have 3 dimensions (sequence_len, batch_size, features)


        # Pack the tensors in obs_seq_resized to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation, -1)

        # Pass the packed sequence through the GRU
        net_out, h = self.rnn(obs_seq_cat, h)

        # Get the last step output of the GRU
        net_out = net_out[:, -1]

        # Process the output from MobileNetV3

        net_out = self.mlp(net_out)  # =====================  tu jest błąd
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.cpu().numpy()

class MobileNetQFunction(nn.Module):
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(np.prod(space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [1], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

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



        # Use adaptive_avg_pool2d to resize the tensors to a fixed size
        max_dim2_size = max(o.shape[2] for o in dupa)
        max_dim3_size = max(o.shape[3] for o in dupa)
        obs_seq_resized = [
            F.adaptive_avg_pool2d(o, output_size=(max_dim2_size, max_dim3_size)) for o in dupa
        ]

        # Transpose the tensors in obs_seq_resized to have the shape (sequence_len, batch_size, features)
        obs_seq_resized = [o.squeeze(dim=3).permute(0, 1, 3, 2) for o in obs_seq_resized]

        # Concatenate the tensors in obs_seq_resized along the second dimension (sequence_len)
        obs_seq_cat = torch.cat(obs_seq_resized, dim=1)

        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = torch.cat((net_out, act), -1)
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)


class MobileNetActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, rnn_size=100,
                 rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()

        # act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedActorMobileNetV3(observation_space, action_space, rnn_size, rnn_len, mlp_sizes,
                                              activation)
        self.q1 = MobileNetQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q2 = MobileNetQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)