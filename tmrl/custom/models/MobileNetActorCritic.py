import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import torch
from torchvision import transforms
import gym
from custom.models.MobileNetV3 import mobilenetv3_large
from custom.models.model_constants import LOG_STD_MIN, LOG_STD_MAX
# Assuming you have already defined `rnn` and other utility functions

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


class ActorCriticMobileNetV3(nn.Module):
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        self.mobilenet = mobilenetv3_large(pretrained=True)  # Load the pre-trained MobileNetV3 model
        self.mobilenet.eval()  # Freeze the pre-trained model

        dim_obs = 1280  # Output feature size from MobileNetV3
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size] + list(mlp_sizes), activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_limit
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    # def extract_features(self, obs):
    #     # Preprocess observations for MobileNetV3
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     processed_obs = torch.cat([transform(o).unsqueeze(0) for o in obs], dim=0)
    #
    #     with torch.no_grad():
    #         features = self.mobilenet(processed_obs)
    #
    #     return features

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        features = self.extract_features(obs_seq)
        features = features.view(features.size(0), -1)

        net_out, h = self.rnn(features, h)
        net_out = net_out[:, -1]
        net_out = self.mlp(net_out)
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
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            if torch.cuda.is_available():
                return a.gpu().numpy()
            else:
                return a.cpu().numpy()
