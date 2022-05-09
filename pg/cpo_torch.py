import pg.agent_torch as torch_agent
import torch
import torch.nn as nn
import math
import numpy as np


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def get_flat_params_from(model):
    # pdb.set_trace()
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def flat_grad(grads, grad_norm=False):
    # swb:返回的梯度是detach过了的啊！！
    flag_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
    if grad_norm:
        with torch.no_grad():
            flag_grad = flag_grad / torch.norm(flag_grad)
    return flag_grad


# swb:这个函数是把entropy转化为std吧！！
def ent_2_std(ent: float) -> float:
    std = math.sqrt(math.e**(2*ent-1)/(2*math.pi))
    return std


# Set up function for computing value loss
def compute_loss_v(data, value, taped=True):
    with torch.set_grad_enabled(taped):
        obs, ret = data['obs'], data['ret']
        v = value(obs)
        loss = (v - ret) ** 2
        loss = loss.mean()
    return loss


# Set up function for computing value loss & cost loss
def compute_loss_v_c(data, value, cost_value, taped=True):
    with torch.set_grad_enabled(taped):
        obs, ret, c_ret = data['obs'], data['ret'], data['cret']
        v = value(obs)
        vc = cost_value(obs)
        loss_v = (v - ret) ** 2
        loss_v_c = (vc - c_ret) ** 2
        loss_v, loss_v_c = loss_v.mean(), loss_v_c.mean()
    return loss_v, loss_v_c


# Set up function for computing PPO policy loss
def compute_loss_pi(data, pi, taped=True):  # swb:单纯的policy-gradient的loss啊！！
    obs, act, adv, log_p_old = data['obs'], data['act'], data['adv'], data['logp']
    with torch.set_grad_enabled(taped):
        # policy loss  swb:pi是正太分布，log_p是一个vector（长度为步数啊！！）
        pi_distribution, log_p = pi(obs, act)  # swb:这个东西输出的是什么啊！？
        ratio = torch.exp(log_p - log_p_old)  # swb：这个是动作的ratio啊！！

        loss_pi = -(ratio * adv).mean()
        ent = pi_distribution.entropy()  # swb:求得熵啊！！
        std1, std2 = ent_2_std(ent[0, 0].item()), ent_2_std(ent[0, 1].item())  # swb:获得两个action的ent啊！！

        # swb:有用的额外信息啊！！(这个计算方法我觉得很有问题啊！！)
        # swb:这个kl的计算，是根据蒙特卡罗采样得到的啊！！(由于数据是在pi_old下采样出来的，所以算平均值的时候，只能是kl(pi_old, pi_new))
        approx_kl = (log_p_old - log_p).mean().item()  # swb:这样计算，就不需要用到老策略的均值和方差了啊！！
    pi_info = dict(approx_kl=approx_kl, ent=ent.mean().item(), std1=std1, std2=std2)
    return loss_pi, pi_info


# Set up function for computing PPO policy loss
def compute_loss_cost_pi(data, pi, taped=True):  # swb:单纯的policy-gradient的loss啊！！
    # TODO：这边需要修改下！！
    # obs, act, cret, log_p_old = data['obs'], data['act'], data['cret'], data['logp']
    obs, act, cret, log_p_old = data['obs'], data['act'], data['cadv'], data['logp']
    with torch.set_grad_enabled(taped):
        pi_distribution, log_p = pi(obs, act)  # swb:这个东西输出的是什么啊！？
        ratio = torch.exp(log_p - log_p_old)  # swb：这个是动作的ratio啊！！
        loss_cost_pi = (ratio * cret).mean()
    return loss_cost_pi


class CPOAgent(object):
    def __init__(self, state_dim: int, action_dim: int, action_type: str, name: str = "CPO", hidden_size: list = None,
                 out_activation=nn.Sigmoid):
        if hidden_size is None:
            hidden_size = [250, 250]
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert action_type in ["continuous", "discrete"]
        self.action_type = action_type
        self.model = torch_agent.MLPActorCritic(observation_dim=state_dim, action_dim=action_dim,
                                                action_type=action_type, hidden_sizes=hidden_size,
                                                output_activation=out_activation, cost_network=True)

    def step(self):
        pass


"""directly compute Hessian*vector from KL"""
def Fvp_direct(v, pi, obs, damping=0.01):
    kl = pi.get_kl(obs)
    kl = kl.mean()
    grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, pi.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()
    return flat_grad_grad_kl + v * damping


def conjugate_gradients(Avp_f, b, pi, obs, nsteps=10, r_norm=1e-10):   # swb:
    x = torch.zeros(b.size(), device=b.device)
    r = b.clone()  # swb:初始化就不太一样啊！！
    p = b.clone()  # swb:初始化就不太一样啊！！
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p, pi, obs)  # swb:这里是A*p
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < r_norm:
            break
    return x


def send_2_worker_and_recv(content, pipes: list):
    length = len(pipes)
    fetch_data_list = []
    for i in range(length):
        pipes[i].send(content)  # swb:发送开始训练的信号！！
    for i in range(length):
        fetch_data_list.append(pipes[i].recv())
    return fetch_data_list


class Data(object):
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            self[key] = item

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)


# swb:把一个dict的list，转化为一个合并所有数据的dict啊！！
def concat_dict_tensor(list_dict: list) -> dict:
    """
    :param list_dict: 输入是一个list，里面都是字典dict啊！！
    :return: 我们返回给用户的是拼接后的数据啊，data为一个dict的形式啊！！
    """
    dict_list = {k: [] for k, _ in list_dict[0].items()}
    for counter in range(len(list_dict)):
        for k, v in list_dict[counter].items():
            dict_list[k].append(v)
    data = dict()
    for k, v in dict_list.items():
        data[k] = torch.cat(v, dim=0)
    return data


# swb:这个函数的输入就是 [[...], [...], [...]] 这种类型的啊！！
def lists_avg(in_list: list):
    assert type(in_list[0]) == list
    buffer = []
    for i in range(len(in_list)):
        buffer.extend(in_list[i])
    summation = sum(buffer)
    if len(buffer) == 0:
        mean = math.nan
    else:
        mean = summation/len(buffer)
    return summation, mean


if __name__ == "__main__":
    pass