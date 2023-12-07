import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

import config.config_constants as cfg
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


def conv2d_out_dims(conv_layer, h_in, w_in):
    if conv_layer.padding == "same":
        return h_in, w_in
    h_out = h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1
    h_out = math.floor(h_out / conv_layer.stride[0] + 1)
    w_out = w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1
    w_out = math.floor(w_out / conv_layer.stride[1] + 1)
    return h_out, w_out


def init_kaiming(layer):
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in")
    torch.nn.init.zeros_(layer.bias)


class CNNModule(nn.Module):
    def __init__(self, mlp_out_size: int = 128,
                 first_out_channels: int = 20, activation=nn.LeakyReLU):
        super(CNNModule, self).__init__()
        self.activation = activation
        self.conv_groups = 2
        self.conv_blocks = nn.ModuleList()
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        self.conv_blocks_len = int(math.log2(self.h_out) - 4)
        first_kernel_size = 5
        first_out_channels = first_out_channels
        hist = cfg.IMG_HIST_LEN
        first_conv = nn.Conv2d(
            hist, first_out_channels, kernel_size=first_kernel_size, stride=1,
            padding=3
        )
        self.h_out, self.w_out = conv2d_out_dims(first_conv, self.h_out, self.w_out)
        self.conv_blocks.append(first_conv)
        self.conv_blocks.append(nn.BatchNorm2d(first_out_channels))
        self.conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.h_out //= 2
        self.w_out //= 2
        self.conv_blocks.append(activation())

        out_channels = first_out_channels

        for _ in range(self.conv_blocks_len):
            prev_out_channels = out_channels
            out_channels = prev_out_channels * 2 - 16

            # Create a residual block
            residual_block = nn.Sequential(
                nn.Conv2d(prev_out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=self.conv_groups),
                nn.BatchNorm2d(out_channels),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=self.conv_groups),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2),
                activation()
            )

            # Add the residual block with a skip connection
            self.conv_blocks.append(residual_block)
            self.h_out //= 2
            self.w_out //= 2

        self.flatten = nn.Flatten()
        out_h, out_w = 4, 12
        flat_features = out_channels * out_h * out_w
        self.mlp_out_size = mlp_out_size
        self.avg_pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        # self.fc1 = nn.Linear(in_features=flat_features, out_features=mlp_out_size)
        self.fc1 = nn.Linear(in_features=flat_features, out_features=mlp_out_size)
        self.dropout = nn.Dropout(0.1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.conv_blocks:
            if isinstance(m, torch.nn.Conv2d):
                init_kaiming(m)
        init_kaiming(self.fc1)

    def forward(self, x):
        # Forward pass through the initial layers
        for layer in self.conv_blocks:
            x = layer(x)

        x = self.avg_pool(x)
        x = self.flatten(x) # 1 x 48 x 4 x 12 => (1, 2304) (256, 2304)
        x = self.fc1(x)
        x = self.dropout(x)

        return x


class QRCNNQFunction(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=192, rnn_len=1, mlp_branch_sizes=(192, 256, 128),
            activation=nn.GELU, num_quantiles=25
    ):
        super().__init__()
        self.conv_branch = CNNModule(mlp_out_size=32)
        self.activation = activation()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_index = 17

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[self.img_index].shape)
        self.num_quantiles = num_quantiles
        dim_act = action_space.shape[0]
        mlp1_size, mlp2_size, mlp3_size = mlp_branch_sizes

        self.layerNormApi = nn.LayerNorm(mlp2_size)
        self.layerNormCNN = nn.LayerNorm(self.conv_branch.mlp_out_size)

        self.mlp1_lvl1 = nn.Linear(dim_obs, mlp1_size)
        self.mlp1_lvl2 = nn.Linear(mlp1_size, mlp2_size)

        self.rnn_blockApi = lstm(
            mlp2_size,
            rnn_size // 2,
            rnn_len
        )

        self.rnn_blockCat = lstm(
            rnn_size // 2 + self.conv_branch.mlp_out_size + dim_act,
            rnn_size,
            rnn_len
        )

        self.noisy_out = NoisyLinear(
            rnn_size,
            mlp3_size,
            device=self.device,
            std_init=0.1
        )

        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, observation, act, save_hidden=False):
        self.rnn_blockApi.flatten_parameters()
        self.rnn_blockCat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        cnn_branch_input = observation[self.img_index].float()
        if batch_size == 1:
            if not cfg.GRAYSCALE:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            # else:
            #     cnn_branch_input = torch.unsqueeze(cnn_branch_input, dim=0)
            conv_branch_out = self.conv_branch(cnn_branch_input)
        else:
            if not cfg.GRAYSCALE:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            # else:
            #     cnn_branch_input = torch.unsqueeze(cnn_branch_input, dim=0)
            #     cnn_branch_input = cnn_branch_input.permute(1, 0, 2, 3)
            conv_branch_out = self.conv_branch(cnn_branch_input)

        # Separate observations at img index
        observation_except_img = observation[:self.img_index]
        if len(observation) > self.img_index + 1:
            observation_except_img += observation[(self.img_index + 1):]

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation_except_img[0].device
            h0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size // 2), device=device)
            )
            c0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size // 2), device=device)
            )
            h1 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
            c1 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
        else:
            h0 = self.h0
            h1 = self.h1
            c0 = self.c0
            c1 = self.c1

        for index, _ in enumerate(observation_except_img):
            observation_except_img[index] = observation_except_img[index].view(batch_size, 1, -1)

        # Pack the tensors in observation_except_24 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        mlp1_lvl1_out = self.activation(self.mlp1_lvl1(obs_seq_cat))
        mlp1_lvl2_out = self.activation(self.mlp1_lvl2(mlp1_lvl1_out))
        layernorm_api_out = self.layerNormApi(mlp1_lvl2_out)
        layernorm_cnn_out = self.layerNormCNN(conv_branch_out)

        rnn_block_api_out, (h0, c0) = self.rnn_blockApi(layernorm_api_out, (h0, c0))

        rnn_api_cnn_cat = torch.cat([rnn_block_api_out, layernorm_cnn_out, act], dim=-1)

        rnn_cat_out, (h1, c1) = self.rnn_blockCat(rnn_api_cnn_cat, (h1, c1))

        noisy_out = self.activation(self.noisy_out(rnn_cat_out))

        # q = self.activation(self.mlp_last(lstm_act_out))

        if save_hidden:
            self.h0 = h0
            self.h1 = h1
            self.c0 = c0
            self.c1 = c1

        return torch.squeeze(noisy_out, -1)


class SquashedActorQRCNN(TorchActorModule):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=192, rnn_len=1, mlp_branch_sizes=(192, 256, 128),
            activation=nn.GELU
    ):
        super().__init__(
            observation_space, action_space
        )
        self.conv_branch = CNNModule(mlp_out_size=32)
        self.activation = activation()
        self.output_activation = nn.Tanh()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        self.img_index = 17
        # dim_obs -= math.prod(s for s in observation_space[24].shape)
        dim_obs -= math.prod(s for s in observation_space[self.img_index].shape)
        dim_act = action_space.shape[0]
        mlp1_size, mlp2_size, mlp3_size = mlp_branch_sizes

        self.layerNormApi = nn.LayerNorm(mlp2_size)
        self.layerNormCNN = nn.LayerNorm(self.conv_branch.mlp_out_size)

        self.mlp1_lvl1 = nn.Linear(dim_obs, mlp1_size)
        self.mlp1_lvl2 = nn.Linear(mlp1_size, mlp2_size)

        self.rnn_blockApi = lstm(
            mlp2_size,
            rnn_size // 2,
            rnn_len
        )

        self.rnn_blockCat = lstm(
            rnn_size // 2 + self.conv_branch.mlp_out_size,
            rnn_size,
            rnn_len
        )

        self.noisy_out = NoisyLinear(
            rnn_size,  # + self.conv_branch2.mlp_out_size,
            mlp3_size,
            device=self.device,
            std_init=0.1
        )

        self.mu_layer = nn.Linear(mlp3_size, dim_act)
        self.log_std_layer = nn.Linear(mlp3_size, dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.squash_correction = 2 * (np.log(2) - np.log(self.act_limit))
        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, observation, test=False, with_logprob=True, save_hidden=False):
        self.rnn_blockApi.flatten_parameters()
        self.rnn_blockCat.flatten_parameters()

        batch_size = observation[0].shape[0]

        if type(observation) is tuple:
            observation = list(observation)
        cnn_branch_input = observation[self.img_index].float()
        if batch_size == 1:
            if not cfg.GRAYSCALE:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            # else:
            #     cnn_branch_input = torch.unsqueeze(cnn_branch_input, dim=0)
            conv_branch_out = self.conv_branch(cnn_branch_input)
        else:
            if not cfg.GRAYSCALE:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            # else:
            #    cnn_branch_input = torch.unsqueeze(cnn_branch_input, dim=0)
            #     cnn_branch_input = cnn_branch_input.permute(1, 0, 2, 3)
            conv_branch_out = self.conv_branch(cnn_branch_input)

            # Separate observations at img index
        observation_except_img = observation[:self.img_index]
        if len(observation) > self.img_index + 1:
            observation_except_img += observation[(self.img_index + 1):]

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation_except_img[0].device
            h0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size // 2), device=device)
            )
            c0 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size // 2), device=device)
            )
            h1 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
            c1 = Variable(
                torch.zeros((self.rnn_len, self.rnn_size), device=device)
            )
        else:
            h0 = self.h0
            h1 = self.h1
            c0 = self.c0
            c1 = self.c1

        for index, _ in enumerate(observation_except_img):
            observation_except_img[index] = observation_except_img[index].view(batch_size, 1, -1)

        # Pack the tensors in observation_except_img to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        mlp1_lvl1_out = self.activation(self.mlp1_lvl1(obs_seq_cat))
        mlp1_lvl2_out = self.activation(self.mlp1_lvl2(mlp1_lvl1_out))
        layernorm_api_out = self.layerNormApi(mlp1_lvl2_out)
        layernorm_cnn_out = self.layerNormCNN(conv_branch_out)

        rnn_block_api_out, (h0, c0) = self.rnn_blockApi(layernorm_api_out, (h0, c0))

        rnn_api_cnn_cat = torch.cat([rnn_block_api_out, layernorm_cnn_out], dim=-1)

        rnn_cat_out, (h1, c1) = self.rnn_blockCat(rnn_api_cnn_cat, (h1, c1))

        # lstm_out, (h0, c0) = self.rnn_block(layernorm_cat, (h0, c0))

        noisy_out = self.output_activation(self.noisy_out(rnn_cat_out))

        mu = self.mu_layer(noisy_out)
        log_std = self.log_std_layer(noisy_out)
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
            self.h1 = h1
            self.c0 = c0
            self.c1 = c1

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
            self, observation_space, action_space, rnn_size=192, rnn_len=1, mlp_branch_sizes=(192, 256, 128),
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
