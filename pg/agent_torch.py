import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # swb：对给定的 observation 产生一个 action 的分布，并且可选地返回此分布下产生输入 action 的 log-likelihood
        pi = self._distribution(obs)  # swb:输出的高斯概率分布啊！！！
        log_p_a = None
        if act is not None:
            # swb:这个东西，就是说给定了观测和动作，看看输出这个动作的概率啊！！
            log_p_a = self._log_prob_from_distribution(pi, act)
        return pi, log_p_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=nn.Identity):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def _distribution(self, obs):
        logit_s = self.logits_net(obs)
        return Categorical(logits=logit_s)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=nn.Identity, std_network=False):
        super().__init__()
        self.std_network = std_network  # swb:这个决定，std是用网络输出，还是就是个parameters啊。
        if std_network:
            self.log_std = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Identity)
        else:
            log_std = -0.5 * np.ones(act_dim, dtype=np.float32)  # swb: log_std = [0.5]
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))  # swb: log_std也是参数啊！！(是可以训练的啊！！)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)  # swb:这边是一个mlp啊！！

    def _distribution(self, obs):
        mu = self.mu_net(obs)  # swb：这边就是网络的输出，动作的均值啊!!!
        if self.std_network:
            log_std = self.log_std(obs)
            std = torch.exp(log_std)
        else:
            std = torch.exp(self.log_std)  # swb：之所以用log_std，是因为这个变化快啊！！
        return Normal(mu, std)  # swb:非常得奇怪啊，为什么这边会有

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def get_std(self):
        if self.std_network is False:
            log_std = self.log_std.data.numpy()  # swb:得到了log-std啊！！
            return np.exp(log_std)  # swb：这样就能得到了std数值了啊！！
        else:
            raise NotImplementedError

    def get_kl(self, x):
        mu1 = self.mu_net(x)
        log_std1, std1 = self.log_std, torch.exp(self.log_std)
        mu0 = mu1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mu0 - mu1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


# TODO:采样的时候，万一超出了阈值，则需要进行clip啊。(不过我看那个mountain-car的环境里面已经对动作进行了clip了啊！！)
# TODO:如果是evaluate的时候，需要有个参数决定，是采样，还是直接输出均值mu。。(执行的时候应该不需要噪音)
# TODO:那个探索的std，是不是应该也拿个网络来更新的啊！？（std应该也用个网络输出，这样std就是和state有关了，感觉上这样会好些）
# TODO:CPU采样，GPU来batch训练啊！！
class MLPActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, action_type: str,
                 hidden_sizes=(64, 64), activation=nn.Tanh, output_activation=nn.Identity,
                 std_network=False, cost_network=False):
        super().__init__()
        assert type(observation_dim) == int and type(action_dim) == int  # swb:合法性检查
        assert action_type in ["continuous", "discrete"]
        # swb:根据动作是离散还是连续，建立不同的policy啊！！
        if action_type == "continuous":
            # swb:这里面说白了两样东西，一个mu值网络和一个std值参数。。
            self.pi = MLPGaussianActor(observation_dim, action_dim, hidden_sizes, activation,
                                       output_activation=output_activation, std_network=std_network)
        elif action_type == "discrete":
            self.pi = MLPCategoricalActor(observation_dim, action_dim, hidden_sizes, activation)
        # swb:建立一个v值网络，用来拟合value function啊！！
        self.v = MLPCritic(observation_dim, hidden_sizes, activation)
        # swb:建立一个vc值网络，用来拟合value function啊！！
        self.cost_network = cost_network
        if cost_network is True:
            self.vc = MLPCritic(observation_dim, hidden_sizes, activation)
        else:
            self.vc = None

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)  # swb：输入 state，输出 action 的分布。
            a = pi.sample()  # swb：根据分布，进行采样。
            log_p_a = self.pi._log_prob_from_distribution(pi, a)  # swb：根据分布和采样，就能计算出采样的似然啊。
            v = self.v(obs)  # swb:根据 state，输出 value
            if self.cost_network:
                vc = self.vc(obs)  # swb:根据 state，输出 cost value
        # swb：action，v值，cost的v值，action的概率
        if self.cost_network is True:
            return a.numpy(), v.numpy(), vc.numpy(), log_p_a.numpy()
        else:
            return a.numpy(), v.numpy(), log_p_a.numpy()

    def act(self, obs, explore=False):
        # swb：这个大概就是 evaluate 的时候，直接使用吧
        # swb:使用的时候，要探索么？！
        if explore:
            return self.step(obs)[0]
        else:
            with torch.no_grad():
                mu = self.pi.mu_net(obs)
                return mu.numpy()







