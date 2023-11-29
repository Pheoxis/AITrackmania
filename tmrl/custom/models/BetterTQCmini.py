import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

import config.config_constants as cfg
import config.config_objects as cfo
from actor import TorchActorModule
from custom.models.model_constants import LOG_STD_MIN, LOG_STD_MAX


# https://discuss.pytorch.org/t/dropout-in-lstm-during-eval-mode/120177
def gru(input_size, rnn_size, rnn_len, dropout: float = 0.1):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru_layers = nn.GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=dropout, bidirectional=False
    )

    return gru_layers


def lstm(input_size, rnn_size, rnn_len, dropout: float = 0.0):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    lstm_layers = nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=dropout, bidirectional=False
    )

    return lstm_layers


class QRCNNQFunction(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=180, rnn_len=2, mlp_branch_sizes=(192, 256, 128),
            activation=nn.GELU
    ):
        super().__init__()
        self.activation = activation()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        mlp1_size, mlp2_size, mlp3_size = mlp_branch_sizes
        self.num_quantiles = cfo.ALG_CONFIG["QUANTILES_NUMBER"]

        self.layerNormApi = nn.LayerNorm(mlp2_size)

        self.mlp1_lvl1 = nn.Linear(dim_obs, mlp1_size)
        self.mlp1_lvl2 = nn.Linear(mlp1_size, mlp2_size)

        self.rnn_blockApi = lstm(
            mlp2_size + dim_act,
            rnn_size,
            rnn_len
        )


        if cfo.ALG_CONFIG["NOISY_LINEAR_CRITIC"]:
            self.model_out = NoisyLinear(
                rnn_size,
                self.num_quantiles,
                device=self.device,
                std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_size, self.num_quantiles)


        self.h0 = None
        self.c0 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, observation, act, save_hidden=False):
        self.rnn_blockApi.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation[0].device
            h0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
            c0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
        else:
            h0 = self.h0
            c0 = self.c0

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        # Pack the tensors in observation_except_24 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1)

        mlp1_lvl1_out = self.activation(self.mlp1_lvl1(obs_seq_cat.float()))
        mlp1_lvl2_out = self.activation(self.mlp1_lvl2(mlp1_lvl1_out))
        layer_norm_api_out = self.layerNormApi(mlp1_lvl2_out)

        cat_layer_norm_act_out = torch.cat([layer_norm_api_out, act], dim=-1)

        rnn_block_api_out, (h0, c0) = self.rnn_blockApi(cat_layer_norm_act_out, (h0, c0))

        model_out = self.model_out(rnn_block_api_out)

        # q = self.activation(self.mlp_last(lstm_act_out))

        if save_hidden:
            self.h0 = h0
            self.c0 = c0

        return torch.squeeze(model_out, -1)


class SquashedActorQRCNN(TorchActorModule):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=180, rnn_len=2, mlp_branch_sizes=(192, 256, 128),
            activation=nn.GELU
    ):
        super().__init__(
            observation_space, action_space
        )
        self.activation = activation()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        mlp1_size, mlp2_size, mlp3_size = mlp_branch_sizes
        mlp_out_size = 1

        self.layerNormApi = nn.LayerNorm(mlp2_size)

        self.mlp1_lvl1 = nn.Linear(dim_obs, mlp1_size)
        self.mlp1_lvl2 = nn.Linear(mlp1_size, mlp2_size)

        self.rnn_blockApi = lstm(
            mlp2_size,
            rnn_size,
            rnn_len
        )

        if cfo.ALG_CONFIG["NOISY_LINEAR_ACTOR"]:
            self.model_out = NoisyLinear(
                rnn_size,
                self.num_quantiles,
                device=self.device,
                std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_size, mlp_out_size)


        self.mu_layer = nn.Linear(mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(mlp_out_size, dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        # self.squash_correction = 2 * (np.log(2) - np.log(self.act_limit))
        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, observation, test=False, with_logprob=True, save_hidden=False):
        self.rnn_blockApi.flatten_parameters()

        batch_size = observation[0].shape[0]

        if type(observation) is tuple:
            observation = list(observation)

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation[0].device
            h0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
            c0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
        else:
            h0 = self.h0
            c0 = self.c0

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        # Pack the tensors in observation_except_img to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        mlp1_lvl1_out = self.activation(self.mlp1_lvl1(obs_seq_cat))
        mlp1_lvl2_out = self.activation(self.mlp1_lvl2(mlp1_lvl1_out))
        layernorm_api_out = self.layerNormApi(mlp1_lvl2_out)

        rnn_block_api_out, (h0, c0) = self.rnn_blockApi(layernorm_api_out, (h0, c0))

        model_out = self.model_out(rnn_block_api_out)

        mu = self.mu_layer(model_out)
        log_std = self.log_std_layer(model_out)
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
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h0 = h0
            self.c0 = c0

        return pi_action, logp_pi

    def act(self, obs: tuple, test=False):
        obs_seq = list(obs)
        # obs_seq = list(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(observation=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.cpu().numpy()


class QRCNNActorCritic(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=180, rnn_len=2, mlp_branch_sizes=(192, 256, 128),
            activation=nn.GELU
    ):
        super().__init__()

        # act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedActorQRCNN(
            observation_space, action_space, rnn_size, rnn_len, mlp_branch_sizes, activation
        )
        self.q1 = QRCNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_branch_sizes, activation
        )
        self.q2 = QRCNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_branch_sizes, activation
        )
