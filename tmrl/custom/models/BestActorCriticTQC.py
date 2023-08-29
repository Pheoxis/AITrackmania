import math

import numpy as np
import torch
import config.config_constants as cfg
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal

from actor import TorchActorModule
from custom.models.model_blocks import mlp
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


def lstm(input_size, rnn_size, rnn_len, dropout: float = 0.1):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    lstm_layers = nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=dropout, bidirectional=False
    )

    return lstm_layers


def conv2d_out_dims(conv_layer, h_in, w_in):
    if conv_layer.padding == "same":
        return h_in, w_in
    h_out = h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1
    h_out = math.floor(h_out / conv_layer.stride[0] + 1)
    w_out = w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1
    w_out = math.floor(w_out / conv_layer.stride[1] + 1)
    return h_out, w_out


class CNNModule(nn.Module):
    def __init__(self, grayscale: bool = cfg.GRAYSCALE, mlp_out_size: int = 256,
                 first_out_channels: int = 16, activation=nn.GELU):
        super(CNNModule, self).__init__()
        self.activation = activation
        # ogarnąć to
        # self.conv_groups = 1 if grayscale else 3
        # self.conv_groups = 1
        self.conv_blocks = nn.ModuleList()
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        self.conv_blocks_len = int(math.log2(self.h_out) - 2)
        first_kernel_size = 6
        first_out_channels = first_out_channels
        first_in_channels = 1 if grayscale else 3
        first_conv = nn.Conv2d(
            first_in_channels, first_out_channels, kernel_size=first_kernel_size, stride=1,
            padding="same"
        )
        self.conv_blocks.append(first_conv)
        self.conv_blocks.append(nn.BatchNorm2d(first_out_channels))
        self.conv_blocks.append(nn.GELU())
        self.h_out, self.w_out = conv2d_out_dims(first_conv, self.h_out, self.w_out)
        out_channels = first_out_channels

        # Create Conv2d and BatchNorm2d layers
        for _ in range(self.conv_blocks_len):
            out_channels *= 2
            next_conv = nn.Conv2d(
                out_channels // 2, out_channels, kernel_size=3, stride=1, padding="same"  # , groups=self.conv_groups
            )
            self.conv_blocks.append(next_conv)
            self.h_out, self.w_out = conv2d_out_dims(next_conv, self.h_out, self.w_out)
            self.conv_blocks.append(nn.BatchNorm2d(out_channels))
            self.conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.h_out //= 2
            self.w_out //= 2
            self.conv_blocks.append(activation())
        self.flatten = nn.Flatten()
        flat_features = out_channels * self.h_out * self.w_out
        self.mlp_out_size = mlp_out_size
        self.fc1 = nn.Linear(in_features=flat_features, out_features=mlp_out_size)

    def forward(self, x):
        # Forward pass through convolutional layers
        for layer in self.conv_blocks:
            x = F.relu(layer(x))  # 1x3x128x256 => 1x512x3x7

        # Flatten the tensor
        x = self.flatten(x)  # (batch size, last conv out channels,

        # Forward pass through MLP layers
        x = self.fc1(x)  # 1x10752

        return x


class QRCNNQFunction(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=128, rnn_len=2, mlp_branch_sizes=(256, 512, 256),
            activation=nn.GELU, num_quantiles=25
    ):
        super().__init__()
        self.conv_branch = CNNModule()
        self.activation = activation()

        # mlp branch
        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[24].shape)
        self.num_quantiles = num_quantiles
        self.layerNorm = nn.LayerNorm(dim_obs)
        mlp1_size, mlp2_size, mlp3_size = mlp_branch_sizes
        self.mlp1 = nn.Linear(dim_obs, mlp1_size)
        self.mlp2 = nn.Linear(mlp1_size, mlp2_size)
        self.mlp3 = nn.Linear(mlp2_size, mlp3_size)
        self.dropoutMlpBranch = nn.Dropout(0.1)

        self.cat_mlp = nn.Linear(mlp3_size + self.conv_branch.mlp_out_size, mlp2_size)
        self.mlp_after_mlp_cat_size = 256
        self.mlp_after_cat = nn.Linear(mlp2_size, self.mlp_after_mlp_cat_size)
        self.rnn_block = lstm(self.mlp_after_mlp_cat_size, rnn_size, rnn_len)
        self.mlp_after_rnn = nn.Linear(rnn_size, self.mlp_after_mlp_cat_size)
        self.mlp_after_rnn2 = nn.Linear(self.mlp_after_mlp_cat_size, mlp3_size)
        self.mlp_after_rnn3 = nn.Linear(mlp3_size, self.conv_branch.mlp_out_size)
        self.q_model_out = nn.Linear(self.conv_branch.mlp_out_size + 3, num_quantiles)
        self.dropoutModelOut = nn.Dropout(0.1)

        self.h0 = None
        self.c0 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, observation, act, save_hidden=False):
        self.rnn_block.flatten_parameters()
        batch_size = observation[0].shape[0]
        conv_branch_out = None
        if type(observation) is tuple:
            observation = list(observation)
        if batch_size == 1:
            cnn_branch_input = observation[24].permute(0, 3, 1, 2).float()
            conv_branch_out = self.conv_branch(cnn_branch_input)
            observation[24] = conv_branch_out
        else:
            cnn_branch_input = observation[24].permute(0, 3, 1, 2).float()
            conv_branch_out = self.conv_branch(cnn_branch_input)
            observation[24] = conv_branch_out
            # observation = list(observation)
            # appended_tensors = []
            # for i, obs in enumerate(observation[24]):
            #     obs = torch.unsqueeze(obs, dim=0)
            #     # cnn_branch_input = obs.permute(0, 3, 1, 2).float()
            #     conv_branch_out = self.conv_branch(obs)
            #     appended_tensors.append(conv_branch_out)
            # appended_tensor = torch.cat(appended_tensors, dim=0)
            # observation[24] = appended_tensor

            # Separate observations at index 24
        observation_except_24 = observation[:24] + observation[25:]

        for index, _ in enumerate(observation_except_24):
            observation_except_24[index] = observation_except_24[index].view(batch_size, 1, -1)

        # Pack the tensors in observation_except_24 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_24, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1)

        layer_norm_out = self.layerNorm(obs_seq_cat)
        mlp1_out = self.activation(self.mlp1(layer_norm_out))
        mlp2_out = self.activation(self.mlp2(mlp1_out))
        mlp3_out = self.activation(self.mlp3(mlp2_out))

        dropout_mlp_branch_out = self.activation(self.dropoutMlpBranch(mlp3_out))
        mlp_branch_cnn_module_cat = torch.cat([dropout_mlp_branch_out, observation[24]], dim=-1)

        cat_mlp_out = self.activation(self.cat_mlp(mlp_branch_cnn_module_cat))
        residual_cat = cat_mlp_out + mlp2_out

        mlp_after_cat_out = self.activation(self.mlp_after_cat(residual_cat))

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation_except_24[0].device
            h = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
            c = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
        else:
            h = self.h0
            c = self.c0

        rnn_block_out, (h, c) = self.rnn_block(mlp_after_cat_out, (h, c))

        mlp_after_rnn_out = self.activation(self.mlp_after_rnn(rnn_block_out))

        residual_rnn_conn = mlp_after_rnn_out + mlp_after_cat_out

        mlp_after_rnn2_out = self.activation(self.mlp_after_rnn2(residual_rnn_conn))

        residual_mlp3_rnn2_out = mlp_after_rnn2_out + mlp3_out

        mlp_after_rnn3_out = self.mlp_after_rnn3(residual_mlp3_rnn2_out)

        residual_mlp_conv = mlp_after_rnn3_out + conv_branch_out

        net_out = torch.cat((residual_mlp_conv, act), -1)

        q = self.q_model_out(net_out)

        q = self.dropoutModelOut(q)

        if save_hidden:
            self.h0 = h
            self.c0 = c

        return torch.squeeze(q, -1)


class SquashedActorQRCNN(TorchActorModule):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=128, rnn_len=2, mlp_branch_sizes=(256, 512, 256),
            activation=nn.GELU
    ):
        super().__init__(
            observation_space, action_space
        )
        self.conv_branch = CNNModule()
        self.activation = activation()

        # mlp branch
        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[24].shape)
        self.layerNorm = nn.LayerNorm(dim_obs)
        mlp1_size, mlp2_size, mlp3_size = mlp_branch_sizes
        self.mlp1 = nn.Linear(dim_obs, mlp1_size)
        self.mlp2 = nn.Linear(mlp1_size, mlp2_size)
        self.mlp3 = nn.Linear(mlp2_size, mlp3_size)
        self.dropoutMlpBranch = nn.Dropout(0.1)

        self.cat_mlp = nn.Linear(mlp3_size + self.conv_branch.mlp_out_size, mlp2_size)
        self.mlp_after_mlp_cat_size = 256
        self.mlp_after_cat = nn.Linear(mlp2_size, self.mlp_after_mlp_cat_size)
        self.rnn_block = lstm(self.mlp_after_mlp_cat_size, rnn_size, rnn_len)
        self.mlp_after_rnn = nn.Linear(rnn_size, self.mlp_after_mlp_cat_size)
        self.mlp_after_rnn2 = nn.Linear(self.mlp_after_mlp_cat_size, mlp3_size)
        self.mlp_after_rnn3 = nn.Linear(mlp3_size, self.conv_branch.mlp_out_size)
        self.dropoutModelOut = nn.Dropout(0.1)

        dim_act = action_space.shape[0]
        self.mu_layer = nn.Linear(self.conv_branch.mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(self.conv_branch.mlp_out_size, dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.squash_correction = 2 * (np.log(2) - np.log(self.act_limit))
        self.h0 = None
        self.c0 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, observation, test=False, with_logprob=True, save_hidden=False):
        self.rnn_block.flatten_parameters()
        batch_size = observation[0].shape[0]
        conv_branch_out = None
        if type(observation) is tuple:
            observation = list(observation)
        if batch_size == 1:
            cnn_branch_input = observation[24].permute(0, 3, 1, 2).float()
            conv_branch_out = self.conv_branch(cnn_branch_input)
            observation[24] = conv_branch_out
        else:
            # appended_tensors = []
            # for i, obs in enumerate(observation[24]):
            #     obs = torch.unsqueeze(obs, dim=0)
            #     cnn_branch_input = obs.permute(0, 3, 1, 2).float()
            #     conv_branch_out = self.conv_branch(cnn_branch_input)
            #     appended_tensors.append(conv_branch_out)
            # appended_tensor = torch.cat(appended_tensors, dim=0)
            # observation[24] = appended_tensor
            cnn_branch_input = observation[24].permute(0, 3, 1, 2).float()
            conv_branch_out = self.conv_branch(cnn_branch_input)
            observation[24] = conv_branch_out

            # Separate observations at index 24
        observation_except_24 = observation[:24] + observation[25:]

        for index, _ in enumerate(observation_except_24):
            observation_except_24[index] = observation_except_24[index].view(batch_size, 1, -1)

        # Pack the tensors in observation_except_24 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_24, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1)

        layer_norm_out = self.layerNorm(obs_seq_cat)
        mlp1_out = self.activation(self.mlp1(layer_norm_out))
        mlp2_out = self.activation(self.mlp2(mlp1_out))
        mlp3_out = self.activation(self.mlp3(mlp2_out))

        dropout_mlp_branch_out = self.activation(self.dropoutMlpBranch(mlp3_out))
        mlp_branch_cnn_module_cat = torch.cat([dropout_mlp_branch_out, observation[24]], dim=-1)

        cat_mlp_out = self.activation(self.cat_mlp(mlp_branch_cnn_module_cat))
        residual_cat = cat_mlp_out + mlp2_out

        mlp_after_cat_out = self.activation(self.mlp_after_cat(residual_cat))

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation_except_24[0].device
            h = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
            c = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
        else:
            h = self.h0
            c = self.c0

        rnn_block_out, (h, c) = self.rnn_block(mlp_after_cat_out, (h, c))

        mlp_after_rnn_out = self.activation(self.mlp_after_rnn(rnn_block_out))

        residual_rnn_conn = mlp_after_rnn_out + mlp_after_cat_out

        mlp_after_rnn2_out = self.activation(self.mlp_after_rnn2(residual_rnn_conn))

        residual_mlp3_rnn2_out = mlp_after_rnn2_out + mlp3_out

        mlp_after_rnn3_out = self.mlp_after_rnn3(residual_mlp3_rnn2_out)

        residual_mlp_conv = mlp_after_rnn3_out + conv_branch_out

        dropout_model_out = self.dropoutModelOut(residual_mlp_conv)

        mu = self.mu_layer(dropout_model_out)
        log_std = self.log_std_layer(dropout_model_out)
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
            self.h0 = h
            self.c0 = c

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
            self, observation_space, action_space, rnn_size=128, rnn_len=2, mlp_branch_sizes=(256, 512, 256),
            activation=nn.GELU, num_quantiles=25
    ):
        super().__init__()

        # act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedActorQRCNN(
            observation_space, action_space, rnn_size, rnn_len, mlp_branch_sizes, activation
        )
        self.q1 = QRCNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_branch_sizes, activation, num_quantiles
        )
        self.q2 = QRCNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_branch_sizes, activation, num_quantiles
        )
