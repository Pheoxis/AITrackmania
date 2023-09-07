import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal

from actor import TorchActorModule

#ok
LOG_STD_MIN = -20
LOG_STD_MAX = 2
# Assuming you have already defined `rnn` and other utility functions
def conv1x1(self, x):
    x = torch.squeeze(x, dim=2)  # Squeeze the third dimension
    x = torch.squeeze(x, dim=2)  # Squeeze the fourth dimension
    return F.conv2d(x, self.conv1x1_weights)

from custom.models.MobileNetV3 import mobilenetv3_large, mobilenetv3_small
from custom.models.model_blocks import mlp
from custom.models.model_constants import LOG_STD_MIN, LOG_STD_MAX



# change this like
def gru(input_size, rnn_size, rnn_len):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru_layers = nn.GRU(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=0, bidirectional=False
    )

    return gru_layers


def lstm(input_size, rnn_size, rnn_len):
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    lstm_layers = nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers,
        bias=True, batch_first=True, dropout=0, bidirectional=False
    )

    return lstm_layers


class MobileNetQFunction(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, act_space, rnn_size=100, rnn_len=2,
            mlp_sizes=(64, 103),
            activation=nn.GELU,
            # activation=nn.ReLU,
            num_classes=8
    ):
        super().__init__()
        # dim_obs = sum(np.prod(space.shape) for space in obs_space)
        # dim_act = act_space.shape[0]
        self.mobilenetQ = mobilenetv3_small(num_classes=num_classes)
        # self.gru = gru(43 + self.mobilenetQ.num_classes, rnn_size, rnn_len)
        self.mlp1_sizes = [43, 128, rnn_size]
        self.bn_rnn = nn.BatchNorm1d(self.mlp1_sizes[0])
        self.mlp1 = mlp(self.mlp1_sizes, activation)
        self.rnn = lstm(self.mlp1_sizes[-1] + self.mobilenetQ.num_classes, rnn_size, rnn_len)
        self.ln1 = nn.LayerNorm(rnn_size)
        self.mlp2 = mlp([rnn_size + 3] + list(mlp_sizes) + [1], activation)
        self.h0 = None
        self.c0 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.dropout = nn.Dropout(0.1)

    def forward(self, observation, act, save_hidden=False):
        self.rnn.flatten_parameters()
        batch_size = observation[0].shape[0]

        if batch_size == 1:
            self.bn_rnn.eval()  # Set batch normalization to evaluation mode
            self.bn_rnn.train(False)
            mobilenet_input = observation[23][0].permute(0, 3, 1, 2).float()
            output = self.mobilenetSquashed(mobilenet_input)
            observation[23] = output
        else:
            self.bn_rnn.train()
            observation = list(observation)
            appended_tensors = []
            for i, obs in enumerate(observation[23]):
                obs = torch.unsqueeze(obs, dim=0)
                mobilenet_input = obs.permute(0, 3, 1, 2).float()
                output = self.mobilenetQ(mobilenet_input)
                appended_tensors.append(output)
            appended_tensor = torch.cat(appended_tensors, dim=0)
            observation[23] = appended_tensor

            # Separate observations at index 23
        observation_except_23 = observation[:23] + observation[24:]

        for index, _ in enumerate(observation_except_23):
            observation_except_23[index] = observation_except_23[index].view(batch_size, 1, -1)

        # Pack the tensors in observation_except_23 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_23, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1)
        # obs_seq_cat = torch.squeeze(obs_seq_cat).float()

        mlp1_out = self.bn_rnn(obs_seq_cat.float())

        mlp1_out = self.mlp1(mlp1_out)

        residual_mlp1_rnn = mlp1_out
        residual_mobile_mlp2 = observation[23]
        # Append the output of mlp1 with observation at index 23
        mlp1_mobile_cat = torch.cat([mlp1_out, observation[23]], dim=-1)

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation_except_23[0].device
            h = Variable(
                torch.zeros((self.rnn_len, self.mlp1_sizes[-1]), device=device)
            )
            c = Variable(
                torch.zeros((self.rnn_len, self.mlp1_sizes[-1]), device=device)
            )
        else:
            h = self.h0
            c = self.c0

        # Pass the packed sequence through the GRU
        net_out, (h, c) = self.rnn(mlp1_mobile_cat, (h, c))

        net_out = net_out + residual_mlp1_rnn
        net_out = self.ln1(net_out)

        # net_out = net_out[:, -1]

        net_out = torch.cat((net_out, act), -1)
        mlp2_out = self.mlp2(net_out)
        q = self.dropout(mlp2_out)

        if save_hidden:
            self.h0 = h
            self.c0 = c

        return torch.squeeze(q, -1)


# class SquashedActorMobileNetV3(nn.Module):  # maybe this will work
class SquashedActorMobileNetV3(TorchActorModule):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=100, rnn_len=2,
            mlp_sizes=(64, 100),
            activation=nn.GELU,
            # activation=nn.ReLU,
            num_classes=8
    ):
        super().__init__(
            observation_space, action_space
        )
        self.mobilenetSquashed = mobilenetv3_large(num_classes=num_classes)
        # dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        # tm = observation_space.spaces
        # dim_obs = dim_obs - prod(observation_space.spaces[3].shape) + self.mobilenet.num_classes
        dim_act = action_space.shape[0]
        # self.gru = gru(43 + self.mobilenetSquashed.num_classes, rnn_size, rnn_len)
        self.mlp1_sizes = [43, 128, 100]
        self.bn_rnn = nn.BatchNorm1d(self.mlp1_sizes[0])
        self.mlp1 = mlp(self.mlp1_sizes, activation)
        self.rnn = lstm(self.mlp1_sizes[-1] + self.mobilenetSquashed.num_classes, rnn_size, rnn_len)
        self.ln1 = nn.LayerNorm(rnn_size)
        self.mlp2 = mlp([rnn_size] + list(mlp_sizes), activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.squash_correction = 2 * (np.log(2) - np.log(self.act_limit))
        self.h0 = None
        self.c0 = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.conv1x1_weights = nn.Parameter(torch.randn(rnn_size, 1, 1))
        self.dropout = nn.Dropout(0.1)

    def conv1x1(self, x):
        return F.conv1d(x, self.conv1x1_weights)

    def forward(self, observation, test=False, with_logprob=True, save_hidden=False):
        # worker: list of 26 elements
        # trainer: tuple of 27 elements of (256, x, y, z)
        self.rnn.flatten_parameters()

        # Convert single observation to a batch if needed
        # if len(observation[0].shape) == 3:
        #    observation = [obs.unsqueeze(0) for obs in observation]

        batch_size = observation[0].shape[0]

        if batch_size == 1:
            self.bn_rnn.eval()  # Set batch normalization to evaluation mode
            self.bn_rnn.train(False)
            mobilenet_input = observation[23][0].permute(0, 3, 1, 2).float()
            output = self.mobilenetSquashed(mobilenet_input)
            observation[23] = output
        else:
            self.bn_rnn.train()
            observation = list(observation)
            appended_tensors = []
            for i, obs in enumerate(observation[23]):
                obs = torch.unsqueeze(obs, dim=0)
                mobilenet_input = obs.permute(0, 3, 1, 2).float()
                output = self.mobilenetSquashed(mobilenet_input)
                appended_tensors.append(output)
            appended_tensor = torch.cat(appended_tensors, dim=0)
            observation[23] = appended_tensor

            # Separate observations at index 23
        observation_except_23 = observation[:23] + observation[24:]

        for index, _ in enumerate(observation_except_23):
            observation_except_23[index] = observation_except_23[index].view(batch_size, 1, -1)

        # Pack the tensors in observation_except_23 to handle variable sequence lengths
        obs_seq_cat = torch.cat(observation_except_23, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1)
        # obs_seq_cat = torch.squeeze(obs_seq_cat).float()

        mlp1_out = self.bn_rnn(obs_seq_cat.float())

        mlp1_out = self.mlp1(mlp1_out)

        residual_mlp1_rnn = mlp1_out
        # Append the output of mlp1 with observation at index 23
        mlp1_mobile_cat = torch.cat([mlp1_out, observation[23]], dim=-1)

        if not save_hidden or self.h0 is None or self.c0 is None:
            device = observation_except_23[0].device
            h = Variable(
                torch.zeros((self.rnn_len, self.mlp1_sizes[-1]), device=device)
            )
            c = Variable(
                torch.zeros((self.rnn_len, self.mlp1_sizes[-1]), device=device)
            )
        else:
            h = self.h0
            c = self.c0

        # Pass the packed sequence through the GRU
        net_out, (h, c) = self.rnn(mlp1_mobile_cat, (h, c))

        net_out = net_out + residual_mlp1_rnn
        net_out = self.ln1(net_out)  # (1, 100)

        # Process the output from MobileNetV3
        net_out = self.mlp2(net_out)
        net_out = self.dropout(net_out)

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

    def act(self, obs, test=False):
        obs_seq = list(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(observation=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.cpu().numpy()


class MobileNetActorCritic(nn.Module):
    # domyślne wartości parametrów muszą się zgadzać
    def __init__(
            self, observation_space, action_space, rnn_size=100, rnn_len=2,
            mlp_sizes=(128, 256, 100),
            activation=nn.GELU,
            # activation=nn.ReLU,
            num_classes=8
    ):
        super().__init__()

        # act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedActorMobileNetV3(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation, num_classes
        )
        self.q1 = MobileNetQFunction(
            action_space, rnn_size, rnn_len, mlp_sizes, activation, num_classes
        )
        self.q2 = MobileNetQFunction(
            action_space, rnn_size, rnn_len, mlp_sizes, activation, num_classes
        )
