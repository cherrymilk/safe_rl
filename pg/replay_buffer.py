import numpy as np
import torch
import scipy.signal


"""
magic from rllab for computing discounted cumulative sums of vectors.
input:  [x0, x1, x2]
output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
"""
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# swb:得到数据的 mean，std，或者 min，max
# swb：不过这个只是用来处理 vector 数据的啊！！
def statistics_scalar(x: np.ndarray, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)
    # swb:确保输入是一个 vector 啊！！
    assert len(x.shape) == 1
    mean = x.mean()
    std = x.std()
    if with_min_and_max:
        max_value = x.max()
        min_value = x.min()
        return mean, std, max_value, min_value
    return mean, std


class ReplayBuffer:
    # swb:这个是我们的replay-buffer啊！！
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        # swb: 我们的这个代码的观测维度和动作维度，必定是1D的，不能是2D的啊！！！！！(之后再升级吧！！)
        assert type(obs_dim) == int and type(act_dim) == int

        # swb:观测和动作的buffer啊！！
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)  # swb:每个观测值是vector啊
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)  # swb:每个动作是vector啊

        # swb: reward！！
        self.rew_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: reward value 啊！！
        self.val_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: reward advantage 啊！！
        self.adv_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊！！

        # swb: 这个是用来计算return的啊！!
        self.ret_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: 这个是记录执行动作的似然概率的啊！！
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        # swb:把每一个 time-step 的环境交互给保存到这里啊！！

    def store(self, obs, act, rew, val, log_p):
        # swb: ptr不能超过buffer的max-size啊！！
        assert self.ptr < self.max_size

        # swb:观测和动作的buffer啊！！
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act

        # swb: reward 啊！！
        self.rew_buf[self.ptr] = rew

        # swb: reward value 啊！！
        self.val_buf[self.ptr] = val

        # swb: 这个是记录执行动作的似然概率的啊！！
        self.logp_buf[self.ptr] = log_p
        self.ptr += 1

    # swb：这个在 一条轨迹结束，或者一条轨迹被中途掐断时 被调用啊！！
    # swb: 这个buffer，会从后往前看。最后一个value，要么是由网络估计出来，或者直接为0。
    def finish_path(self, last_val=0):
        # swb: self.ptr 这个会随着添加buffer，而进行 +1 啊！！
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val)  # swb:这个就是一条 trajectory 的 reward 啊！！

        vals = np.append(self.val_buf[path_slice], last_val)  # swb:这个就是一条 trajectory 的 value 啊！！

        # the next line implements GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]  # swb:gae中的deltas啊！！

        # swb:这个就是 advantage 啊！！
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        # swb:这个就是计算 value 值和 cost value 啊！！！
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    # swb:这个是每个epoch，汇总完所有的经验后，就执行啊！！
    # swb:并会对buffer的指针进行清理啊！！
    # swb:并且，是不是要对数据做归一化，自己决定。
    def get(self):
        assert self.ptr == self.max_size  # swb:在获得数据前，数据必须要填满啊！！
        self.ptr, self.path_start_idx = 0, 0  # swb：重置指针为0啊！！
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        # swb:最后我们把 advantage 给归一化一下啊！！
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # swb:数据用字典封装了！！
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class ReplayBufferWithCost:
    # swb:这个是我们的replay-buffer啊！！
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        # swb: 我们的这个代码的观测维度和动作维度，必定是1D的，不能是2D的啊！！！！！(之后再升级吧！！)
        assert type(obs_dim) == int and type(act_dim) == int

        # swb:观测和动作的buffer啊！！
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)  # swb:每个观测值是vector啊
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)  # swb:每个动作是vector啊

        # swb: reward 和 cost 啊！！
        self.rew_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊
        self.crew_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: reward value 和 cost value 啊！！
        self.val_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊
        self.cval_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: reward advantage 和 cost advantage 啊！！
        self.adv_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊！！
        self.cadv_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: 这个是用来计算return的啊！!
        self.ret_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊
        self.cret_buf = np.zeros(size, dtype=np.float32)  # swb:这些都是标量，所以用个 vector 就可以了啊

        # swb: 这个是记录执行动作的似然概率的啊！！
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        # swb:把每一个 time-step 的环境交互给保存到这里啊！！
    def store(self, obs, act, rew, crew, val, cval, logp):
        # swb: ptr不能超过buffer的max-size啊！！
        assert self.ptr < self.max_size

        # swb:观测和动作的buffer啊！！
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act

        # swb: reward 和 cost 啊！！
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        # swb: reward value 和 cost value 啊！！
        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        # swb: 这个是记录执行动作的似然概率的啊！！
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    # swb：这个在 一条轨迹结束，或者一条轨迹被中途掐断时 被调用啊！！
    # swb: 这个buffer，会从后往前看。最后一个value，要么是由网络估计出来，或者直接为0。
    def finish_path(self, last_val=0, last_cval=0):
        # swb: self.ptr 这个会随着添加buffer，而进行 +1 啊！！
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val)  # swb:这个就是一条 trajectory 的 reward 啊！！
        crews = np.append(self.crew_buf[path_slice], last_cval)  # swb:这个就是一条 trajectory 的 cost 啊！！

        vals = np.append(self.val_buf[path_slice], last_val)  # swb:这个就是一条 trajectory 的 value 啊！！
        cvals = np.append(self.cval_buf[path_slice], last_cval)  # swb:这个就是一条 trajectory 的 cost value 啊！！

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]  # swb:gae中的deltas啊！！
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]  # swb:gae中的deltas啊！！

        # swb:这个就是 advantage 啊！！
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        # swb:这个就是计算 value 值和 cost value 啊！！！
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    # swb:这个是每个epoch，汇总完所有的经验后，就执行啊！！
    # swb:并会对buffer的指针进行清理啊！！
    # swb:并且，是不是要对数据做归一化，自己决定。
    def get(self):
        assert self.ptr == self.max_size  # swb:在获得数据前，数据必须要填满啊！！
        self.ptr, self.path_start_idx = 0, 0  # swb：重置指针为0啊！！
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = statistics_scalar(self.cadv_buf)
        # swb:最后我们把 advantage 给归一化一下啊！！
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean) / cadv_std   # swb:当时我就很好奇，为什么这边不归一化呢！？

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf,
                    adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


if __name__ == "__main__":
    buffer = ReplayBuffer(obs_dim=4, act_dim=2, size=10)
    for i in range(10):
        buffer.store(np.random.rand(4), np.random.rand(2), np.random.rand(), np.random.rand(), np.random.rand())
    buffer.finish_path(0)
    data = buffer.get()
    print("Test Finished!")
