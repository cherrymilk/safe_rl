import gym
import safety_gym
from pg.cpo_torch import *
import torch
from torch.optim import Adam
from utils.run_utils import setup_logger_path
from pg.logger import EpochLogger
import torch.nn as nn
import numpy as np
from pg.replay_buffer import ReplayBufferWithCost
import time
import math
from multiprocessing import Process
from multiprocessing import Pipe
import multiprocessing as mp
import random
import pdb
EPS = 1e-8


def update(ac, mean_cp_cost, data, cost_limit, vf_optimizer, cvf_optimizer, epoch_num, target_kl, v_iterations, logger,
           train_device, return_device, rescale, grad_norm=False):
    # swb:把数据移动到对应的设备上啊！！ (penalty_param不用移动哦！！)
    if train_device != "cpu":
        ac.to(train_device)
        # swb:这个to函数
        for k, item in data.items():
            data[k] = data[k].to(train_device)
    # swb:先把旧网络的 policy loss 和 value loss 都拿到手，然后看更新后降低了多少！！
    pi_l_old, pi_info_old = compute_loss_pi(data=data, pi=ac.pi, taped=False)
    pi_l_c_old = compute_loss_cost_pi(data=data, pi=ac.pi, taped=False)
    v_l_old, vc_l_old = compute_loss_v_c(data, value=ac.v, cost_value=ac.vc, taped=False)
    # swb: value 和 policy 网络的loss拿到手了啊！！
    pi_l_old, pi_l_c_old, v_l_old, vc_l_old = pi_l_old.item(), pi_l_c_old.item(), v_l_old.item(), vc_l_old.item()
    # swb:先更新简单value网络吧！！
    # swb:价值函数的学习啊！！

    for j in range(v_iterations):
        loss_v, loss_v_c = compute_loss_v_c(data=data, value=ac.v, cost_value=ac.vc)
        vf_optimizer.zero_grad()
        loss_v.backward()
        vf_optimizer.step()
        cvf_optimizer.zero_grad()
        loss_v_c.backward()
        cvf_optimizer.step()
    v_l_new, vc_l_new = loss_v.item(), loss_v_c.item()  # swb:最新的loss啊！！
    # swb:然后开始计算复杂的cpo了啊！！
    cost_deviate = (mean_cp_cost - cost_limit)
    cost_deviate /= rescale
    # step-1 --> 计算 g & H^-1*g
    pi_l, _ = compute_loss_pi(data=data, pi=ac.pi)
    g = flat_grad(torch.autograd.grad(pi_l, ac.pi.parameters()), grad_norm=grad_norm)  # swb:获得了g
    H_inv_g = conjugate_gradients(Avp_f=Fvp_direct, b=g, pi=ac.pi, obs=data['obs'], nsteps=10)

    # step-2 --> 计算 b & H^-1*b
    pi_c_l = compute_loss_cost_pi(data=data, pi=ac.pi)
    b = flat_grad(torch.autograd.grad(pi_c_l, ac.pi.parameters()), grad_norm=grad_norm)  # swb:allow_unused=True的话，无关情况下返回None
    H_inv_b = conjugate_gradients(Avp_f=Fvp_direct, b=b, pi=ac.pi, obs=data['obs'], nsteps=10)

    if torch.norm(b).item()**2 <= 1e-8 and cost_deviate < 0:  # swb：如果对cost的梯度很小或者约束已经满足了，那么就不需要考虑它们了啊！！
        # feasible and cost grad is zero---shortcut to pure TRPO update!
        w, r, s, A, B = 0, 0, 0, 0, 0
        optim_case = 4
    else:
        w, v = H_inv_b, H_inv_g
        q = g.dot(H_inv_g).item()  # g^T*H^{-1}*g
        r = g.dot(H_inv_b).item()  # g^T*H^{-1}*b
        s = b.dot(H_inv_b).item()  # b^T*H^{-1}*b
        A = q - r ** 2 / s  # should be always positive (Cauchy-Schwarz)
        B = 2 * target_kl - cost_deviate ** 2 / s
        if cost_deviate < 0 and B < 0:
            # point in trust region is feasible and safety boundary doesn't intersect
            # ==> entire trust region is feasible
            optim_case = 3  # swb:对的，这种情况下，的确是不用考虑那个约束了啊！！我觉得这个很少会进来的吧！！
        elif cost_deviate < 0 and B >= 0:
            # x = 0 is feasible and safety boundary intersects
            # ==> most of trust region is feasible
            optim_case = 2
        elif cost_deviate >= 0 and B >= 0:
            # x = 0 is infeasible and safety boundary intersects
            # ==> part of trust region is feasible, recovery possible
            optim_case = 1
            print(colorize(f'Epoch:{epoch_num}, Alert! Attempting feasible recovery!', 'yellow'))
        else:
            # x = 0 infeasible, and safety halfspace is outside trust region
            # ==> whole trust region is infeasible, try to fail gracefully
            optim_case = 0
            print(colorize(f'Epoch:{epoch_num}, Alert! Attempting infeasible recovery!', 'red'))

    if optim_case in [3, 4]:
        lam = math.sqrt(q / (2 * target_kl))  # swb:最大化reward，同时满足kl散度的约束！！
        nu = 0
    elif optim_case in [1, 2]:
        LA, LB = [0, r / cost_deviate], [r / cost_deviate, np.inf]
        LA, LB = (LA, LB) if cost_deviate < 0 else (LB, LA)
        proj = lambda x, L: max(L[0], min(L[1], x))
        lam_a = proj(math.sqrt(A / B), LA)
        lam_b = proj(math.sqrt(q / (2 * target_kl)), LB)
        f_a = lambda lam: -0.5 * (A / (lam + EPS) + B * lam) - r * cost_deviate / (s + EPS)
        f_b = lambda lam: -0.5 * (q / (lam + EPS) + 2 * target_kl * lam)
        lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
        nu = max(0, lam * cost_deviate - r) / (s + EPS)
    else:
        lam = 0
        nu = math.sqrt(2 * target_kl / (s + EPS))  # swb:这个公式的意义是说，如果不满足约束的话，就会最小化cost，同时满足散度的约束啊！！

    # normal step if optim_case > 0, but for optim_case =0,
    # perform infeasible recovery: step to purely decrease cost  swb:单纯地降低cost？！我为什么没有看出来啊！？
    x = (1. / (lam + EPS)) * (H_inv_g + nu * w) if optim_case > 0 else nu * w

    pre_param = get_flat_params_from(ac.pi)
    new_param = pre_param - x
    set_flat_params_to(ac.pi, new_param)

    pi_l_new, pi_info_new = compute_loss_pi(data=data, pi=ac.pi, taped=False)
    pi_l_c_new = compute_loss_cost_pi(data=data, pi=ac.pi, taped=False)
    pi_l_new, pi_l_c_new = pi_l_new.item(), pi_l_c_new.item()
    print(colorize(string=f"Epoch:{epoch_num}, curr-cost: {round(mean_cp_cost, 2)}, cost-limit: {round(cost_limit, 2)}, "
                          f"initial-surr-cost: {round(pi_l_c_old, 4)}, after-surr-cost: {round(pi_l_c_new, 4)}", color="cyan"))
    # swb:然后做一些记录吧！！
    # swb:这部分都是和value有关的东西啊！！
    logger.tf_board_plot(key="DeltaLossV", value=v_l_new-v_l_old, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="DeltaLossVC", value=vc_l_new-vc_l_old, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="LossV", value=v_l_new, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="LossVC", value=vc_l_new, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="Penalty", value=nu, iterations=epoch_num,
                         category="1_train_log")

    # swb：以下都是和policy有关的metric啊！！
    logger.tf_board_plot(key="LossPi", value=pi_l_new, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="DeltaLossPi", value=pi_l_new-pi_l_old, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="SurrCost", value=pi_l_c_new, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="DeltaSurrCost", value=pi_l_c_new-pi_l_c_old, iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="Entropy", value=pi_info_new["ent"], iterations=epoch_num,
                         category="1_train_log")
    logger.tf_board_plot(key="KL", value=pi_info_new["approx_kl"], iterations=epoch_num,
                         category="1_train_log")
    # swb:最后记得把模型给还原回去啊！！
    ac.to(return_device)


class WorkerTrain(Process):
    def __init__(self, process_id: int, seed: int, c_pipe: Pipe):
        super().__init__()
        self.process_id = process_id
        self.child_pipe = c_pipe
        self.seed = seed

    def run(self):
        env_p, agent_p, buf_p, local_steps, max_ep_len = None, None, None, None, None
        sample_device = None
        while True:
            content = self.child_pipe.recv()  # swb:收到消息，看看
            status = content.status
            if status == "start_train":
                pass
            elif status == "update_model":
                model_param = content.model_param
                set_flat_params_to(agent_p.model, model_param.to(sample_device))
                self.child_pipe.send("updated!")
                continue
            elif status == "first_init":
                gym.logger.set_level(40)
                env_p = gym.make(content.env_name)
                env_p.seed(self.seed)
                torch.manual_seed(seed=self.seed)
                np.random.seed(self.seed)
                random.seed(self.seed)
                ob = env_p.reset()
                sample_device = content.sample_device
                agent_p = CPOAgent(state_dim=content.state_dim, action_dim=content.action_dim,
                                   action_type=content.action_type, hidden_size=content.hidden_size,
                                   out_activation=eval(content.out_activation))
                agent_p.model.to(sample_device)
                buf_p = ReplayBufferWithCost(content.state_dim, content.action_dim,
                                             content.local_epoch_steps, content.gam, content.lam)
                local_steps = content.local_epoch_steps
                max_ep_len = content.max_ep_len
                self.child_pipe.send("inited!")
                print(f"Process {self.process_id}, init finished!!")
                continue
            elif status == "stop_train":
                print(colorize(f"Process {self.process_id}, sampling finished!!", "yellow"))
                self.child_pipe.send("finished!")
                break  # swb:直接结束啊！！
            else:
                raise NotImplementedError
            # swb：开始写采样的代码吧！？！
            ep_ret_list, ep_cost_list, ep_len_list = [], [], []
            ep_ret, ep_cost, ep_len = 0, 0, 0
            for t in range(local_steps):
                a, v, vc, log_p = agent_p.model.step(torch.as_tensor(ob, dtype=torch.float32).to(sample_device))
                ob_next, r, d, info = env_p.step(a)
                c = info.get('cost', 0)
                # Track cumulative cost over training
                buf_p.store(obs=ob, act=a, rew=r, crew=c, val=v, cval=vc, logp=log_p)
                ep_ret += r  # swb:一个回合的收益
                ep_cost += c  # swb：一个回合的损失
                ep_len += 1  # swb：一个回合的长度
                ob = ob_next  # swb: o_{t} = o_{t+1}
                terminal = (d or (ep_len == max_ep_len))
                if terminal or (t == local_steps - 1):
                    if d and not (ep_len == max_ep_len):  # swb：不是由于episode过长导致的结束
                        last_val, last_cval = 0, 0  # swb：还没到最大回合长度就结束了，所以最后的value和cost都是0啊。。
                    else:
                        _, last_val, last_cval, _ = agent_p.model.step(
                            torch.as_tensor(ob, dtype=torch.float32).to(sample_device))
                    # swb:这边去求一些gae啊有关的东西！！(反正，一个episode结束之后，计算次GAE啊)
                    buf_p.finish_path(last_val, last_cval)
                    # Only save EpRet / EpLen if trajectory finished
                    if terminal is True:
                        ep_ret_list.append(ep_ret)
                        ep_cost_list.append(ep_cost)
                        ep_len_list.append(ep_len)
                    # swb:之后，我们就重置环境吧
                    # Reset environment
                    ob, r, d, c, ep_ret, ep_len, ep_cost = env_p.reset(), 0, False, 0, 0, 0, 0
            # swb:一个epoch的采样结束了啊！！返回吧！！
            avg_r = float(np.mean(buf_p.rew_buf))  # swb:把reward给均值一下吧！！
            buffer_data = buf_p.get()
            data = Data(average_r=avg_r, data_dict=buffer_data, ep_ret_list=ep_ret_list,
                        ep_cost_list=ep_cost_list, ep_len_list=ep_len_list)
            self.child_pipe.send(data)


def process_train_fetch_data_list(fetch_data_list: list, logger, iteration: int) -> dict:
    data = concat_dict_tensor([each.data_dict for each in fetch_data_list])
    avg_r = sum([each.average_r for each in fetch_data_list])/len([each.average_r for each in fetch_data_list])
    _, avg_ep_ret = lists_avg([each.ep_ret_list for each in fetch_data_list])
    _, ep_cost_list = lists_avg([each.ep_cost_list for each in fetch_data_list])
    _, ep_len_list = lists_avg([each.ep_len_list for each in fetch_data_list])
    # swb：记录一些东西啊！！
    logger.tf_board_plot(key="AverageEpRet", value=avg_ep_ret, iterations=iteration,
                         category="1_train_log")
    logger.tf_board_plot(key="AverageEpCost", value=ep_cost_list, iterations=iteration,
                         category="1_train_log")
    data_return = dict()
    data_return['data'] = data
    data_return['avg_r'] = avg_r
    data_return["ep_cost_list"] = ep_cost_list
    return data_return


def main(robot, task, seed, num_steps, steps_per_epoch,
         critic_lr, v_iter, gam, lam, target_kl, sample_device, training_device,
         exp_name, max_ep_len, num_worker, cost_limit, rescale):
    # swb:先设置好随机数种子啊！！
    mp.set_start_method("spawn")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    epochs = int(num_steps / steps_per_epoch)
    # Prepare Logger
    logger_url = setup_logger_path(exp_name=exp_name, data_dir="./data/",
                                   datestamp=True, env_num=num_worker,
                                   seed=seed, rescale=rescale)
    logger = EpochLogger(logger_url)

    # TODO：创建workers(local-steps, seed, agent)
    workers, parent_pipes = [], []
    for counter in range(num_worker):
        process_seed = seed+counter+1
        p_pipe, c_pipe = Pipe()
        worker = WorkerTrain(process_id=counter+1, seed=process_seed, c_pipe=c_pipe)
        workers.append(worker)
        parent_pipes.append(p_pipe)
    for counter in range(len(workers)):
        workers[counter].start()
    # swb:初始化worker吧！！
    ob_dim, a_dim, a_type, hid_size, out_activation = 60, 2, "continuous", [150, 100, 100], "nn.Tanh"
    send_2_worker_and_recv(Data(status="first_init",
                                env_name='Safexp-' + robot + task + '-v0',
                                local_epoch_steps=int(steps_per_epoch / num_worker),
                                gam=gam, lam=lam,
                                hidden_size=hid_size,
                                out_activation=out_activation,
                                state_dim=ob_dim,
                                action_dim=a_dim,
                                action_type=a_type,
                                max_ep_len=max_ep_len,
                                sample_device=sample_device), parent_pipes)

    agent = CPOAgent(state_dim=ob_dim, action_dim=a_dim,
                     action_type=a_type, hidden_size=hid_size,
                     out_activation=eval(out_activation))

    # swb：算法，我也不知道动作输出应该是怎样子的，所以就直接这样子吧！！
    vf_optimizer = Adam(agent.model.v.parameters(), lr=critic_lr)
    cvf_optimizer = Adam(agent.model.vc.parameters(), lr=critic_lr)
    send_2_worker_and_recv(Data(status="update_model", model_param=get_flat_params_from(agent.model).detach()),
                           parent_pipes)

    for epoch in range(epochs):
        start_time = time.time()
        fetch_data_list = send_2_worker_and_recv(Data(status="start_train"), parent_pipes)
        data_dict = process_train_fetch_data_list(fetch_data_list, logger, epoch + 1)
        curr_cost, data_for_train = data_dict["ep_cost_list"], data_dict["data"]
        end_time = time.time()
        sampling_time = round(end_time - start_time, 3)  # swb:采样所消耗的时间啊！！
        # =====================================================================#
        #  Run RL update                                                       #
        # =====================================================================#
        update_start = time.time()  # swb:一次更新开始啊
        update(ac=agent.model, mean_cp_cost=curr_cost, cost_limit=cost_limit, data=data_dict["data"],
               vf_optimizer=vf_optimizer, cvf_optimizer=cvf_optimizer, epoch_num=epoch+1, target_kl=target_kl,
               v_iterations=v_iter, logger=logger, train_device=training_device, return_device=sample_device,
               rescale=rescale)
        torch.save(agent.model.state_dict(), "{}/model.pt".format(logger_url))
        send_2_worker_and_recv(Data(status="update_model", model_param=get_flat_params_from(agent.model).detach()),
                               parent_pipes)
        update_end = time.time()  # swb:一次更新完毕啊
        training_time = round(update_end - update_start, 3)
        print(f"Epoch:{epoch+1}, Sampling consumes {sampling_time} (s), "
              f"Updating consumes {training_time} (s).")
    # swb:结束训练了啊！！
    send_2_worker_and_recv(Data(status="stop_train"), parent_pipes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('parse configuration file')
    parser.add_argument('--robot', type=str, default='Point')  # 'point', 'car', 'doggo'
    parser.add_argument('--task', type=str, default='Goal1')  # 'goal1', 'goal2', 'button1', 'button2', 'push1', 'push2'
    parser.add_argument('--seed', type=int, default=0)  # 0, 10, 20
    parser.add_argument('--gpu_sample', default=False, action='store_true')
    parser.add_argument('--gpu_train', default=True, action='store_true')
    parser.add_argument('--critic_learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--train_v_iterations', type=int, default=80)
    parser.add_argument('--max_ep_len', type=int, default=1000)  # swb:一个episode最大的允许长度啊！！
    parser.add_argument("--ent_reg", type=float, default=0.000)  # swb:这个是entropy的
    parser.add_argument("--cost_limit", type=float, default=25)  # swb:把这个控制在0.07吧！！(碰撞率啊)
    parser.add_argument("--rescale", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)  # swb:[1,2,3,5,6,10,12,15,30]
    parser.add_argument("--steps_per_epoch", type=int, default=30000)
    parser.add_argument("--total_steps", type=int, default=1e5)
    args = parser.parse_args()
    device_sample = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_sample else "cpu")
    device_train = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_train else "cpu")
    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    assert args.task.lower() in task_list, "Invalid task"
    assert args.robot.lower() in robot_list, "Invalid robot"
    main(robot=args.robot, task=args.task, seed=args.seed, num_steps=args.total_steps,
         steps_per_epoch=args.steps_per_epoch, critic_lr=args.critic_learning_rate, v_iter=args.train_v_iterations,
         gam=args.gamma, lam=args.lam, target_kl=args.target_kl, sample_device=device_sample,
         training_device=device_train, exp_name="cpo" + '_' + args.robot.capitalize() + args.task.capitalize(),
         max_ep_len=args.max_ep_len, num_worker=args.num_workers, cost_limit=args.cost_limit,
         rescale=args.rescale)










