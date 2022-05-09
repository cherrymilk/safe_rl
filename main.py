import gym
from pg.algos import ppo, ppo_lagrangian, trpo, trpo_lagrangian, cpo
import safety_gym
from utils.run_utils import setup_logger_kwargs
from utils.mpi_tools import mpi_fork


def main(robot, task, algo, seed, exp_name, cpu):
    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"
    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot == 'Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)  # swb:如果只是1的话，就不用考虑了啊！！(fork有分叉的意思啊，所以有点形象？)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir="./data/")
    # '~/work/code/safe_procject/safety-starter-agents/data/2022-03-21_ppo_lagrangian_PointGoal1/2022-03-21_11-47-35-ppo_lagrangian_PointGoal1_s0'
    # Algo and Env
    algo = eval(algo)  # eval的作用就是执行这条字符串呢。。
    env_name = 'Safexp-'+robot+task+'-v0'
    print('Testing start')
    algo(env_fn=lambda: gym.make(env_name),  #
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    print("SHABI")
    parser.add_argument('--robot', type=str, default='Point')  # 'point', 'car', 'doggo'
    parser.add_argument('--task', type=str, default='Goal1')  # 'goal1', 'goal2', 'button1', 'button2', 'push1', 'push2'
    parser.add_argument('--algo', type=str, default='cpo')  # 'ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo'
    parser.add_argument('--seed', type=int, default=0)  # 0, 10, 20
    parser.add_argument('--exp_name', type=str, default='swb_save')  # 决定保存在哪里啊
    parser.add_argument('--cpu', type=int, default=4)  # 决定采取多少的并行化啊！！
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name == ' ') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu)




