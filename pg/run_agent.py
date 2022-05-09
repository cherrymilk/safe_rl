import numpy as np
import tensorflow as tf
import gym
import time
import pg.trust_region as tro
from pg.agents import PPOAgent, TRPOAgent, CPOAgent
from pg.buffer import CPOBuffer
from pg.network import count_vars, \
                               get_vars, \
                               mlp_actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from pg.utils import values_as_sorted_list
from utils.logx import EpochLogger
from utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum, msg


# Multi-purpose agent runner for policy optimization algos
# (PPO, TRPO, their primal-dual equivalents, CPO)
def run_polopt_agent(env_fn,
                     agent=PPOAgent(),  # swb：这个agent已经有人传过来了，所以不用这个啊！！
                     actor_critic=mlp_actor_critic,  # ()
                     ac_kwargs=dict(),  # (256, 256)
                     seed=0,
                     render=False,
                     # Experience collection:
                     steps_per_epoch=4000,
                     epochs=50,
                     max_ep_len=1000,
                     # Discount factors:
                     gamma=0.99,
                     lam=0.97,  # swb：这个是什么啊！？
                     cost_gamma=0.99,
                     cost_lam=0.97,  # swb：这个是什么啊！？
                     # Policy learning:
                     ent_reg=0.,  # swb:对熵进行惩罚啊
                     # Cost constraints / penalties:
                     cost_lim=25,
                     penalty_init=1.,
                     penalty_lr=5e-2,
                     # KL divergence:
                     target_kl=0.01,
                     # Value learning:
                     vf_lr=1e-3,
                     vf_iters=80,
                     # Logging:
                     logger=None,
                     logger_kwargs=dict(),
                     save_freq=1
                     ):
    # =========================================================================#
    #  Prepare logger, seed, and environment in this process                   #
    # =========================================================================#
    # swb:'output_dir':'/home/wenbin/work/code/safe_procject/safety-starter-agents/data/2022-03-22_ppo_lagrangian_PointGoal1/2022-03-22_01-08-12-ppo_lagrangian_PointGoal1_s0'
    # swb:注意，只有0号脚本，才会建立output_dir和output_file来保存信息啊！！
    logger = EpochLogger(**logger_kwargs, tf_board_logger=True) if logger is None else logger
    logger.save_config(locals())  # swb:这边会把各种配置给打印出来同时保存成文件。。
    # swb：locals()函数会以字典的形式返回当前位置的全部局部变量。。
    seed += 10000 * proc_id()  # swb:确保每个线程之间的采样会各不相同呢。。（proc_id就是MPI的每个worker的ID号码啊！）
    tf.set_random_seed(seed)  # swb:赋予随机数种子
    np.random.seed(seed)
    # swb：每个worker都拥有独立的环境啊
    env = env_fn()  # swb:这边就相当于执行了 gym.make("Safexp-PointGoal1-v0")

    agent.set_logger(logger)  # swb:给agent赋予了logger对象啊。。

    # =========================================================================#
    #  Create computation graph for actor and critic (not training routine)   #
    # =========================================================================#

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space  # swb:Box(2,)
    # swb：简而言之，每个placeholer可以看作都是一个待定的输入啊，相当于一个tensor的抽头
    # Inputs to computation graph from environment spaces
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    # swb:x_place_holder(输入状态) and a_place_holder(输出动作)，两个占位符啊！！
    # Inputs to computation graph for batch data
    adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

    # Inputs to computation graph for special purposes(swb:这个干什么用的啊？!)
    # surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
    cur_cost_ph = tf.placeholder(tf.float32, shape=())  # swb:这是一个实数
    # Outputs from actor critic
    ac_outs = actor_critic(x_ph, a_ph, **ac_kwargs)
    # swb:结果一个ac网络具有这么多东西啊！！！！！！！(pi_info_phs是旧策略的均值和方差的placeholders)
    # swb：pi是采样出的动作，logp是采样a这个动作的概率(依赖a)，logp—pi是采样出pi这个动作的概率，
    # swb：pi_info是当前的动作信息，pi_info_phs是旧网络的动作信息的占位符，d_kl是散度(依赖于状态和之前老的策略)，ent是当前的熵，v和vc都是及时的值输出。
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs
    # swb:所以交互一次要保存的东西有pi，logp_pi，pi_info，v，vc (v和vc是为了计算那个GAE啊！！)

    # Organize placeholders for zipping with data from buffer on updates
    buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
    buf_phs += values_as_sorted_list(pi_info_phs)
    # swb:这里buf_phs里面有9个占位符，所以到时候需要9个东西填补进来啊！
    # Organize symbols we have to compute at each step of acting in env swb:以下这些东西，每步都要计算的啊。。
    get_action_ops = dict(pi=pi,  # swb:这些东西都要留着的啊！！
                          v=v,
                          logp_pi=logp_pi,
                          pi_info=pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    if not (agent.reward_penalized):  # swb:reward_penalized是说把惩罚加入到reward里面啊
        get_action_ops['vc'] = vc  # swb：因为要额外考虑约束，所以分开来了啊！！

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n' % var_counts)
    # swb:上面这个只有0号脚本才会输出啊，只是单纯地打印输出到命令行里面啊！！
    # Make a sample estimate for entropy to use as sanity check
    # swb:用采样来近似entropy，这个是来进行校验的啊？！(没地方使用)
    # approx_ent = tf.reduce_mean(-logp)

    # =========================================================================#
    #  Create replay buffer                                                   #
    # =========================================================================#

    # Obs/act shapes
    obs_shape = env.observation_space.shape  # swb: 60
    act_shape = env.action_space.shape  # swb：2

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())  # swb:一个epoch要收集到这么多的经验条才行啊！！！
    # swb:这个是在记录 mu 和 std 的纬度啊！！！
    pi_info_shapes = {k: v.shape.as_list()[1:] for k, v in pi_info_phs.items()}
    buf = CPOBuffer(local_steps_per_epoch,
                    obs_shape,
                    act_shape,
                    pi_info_shapes,
                    gamma,
                    lam,
                    cost_gamma,
                    cost_lam)  # swb：lam是用来干什么的？？

    # =========================================================================#
    #  Create computation graph for penalty learning, if applicable           #
    # =========================================================================#
    # swb:哦，这就是那个惩罚系数么？！！！
    if agent.use_penalty:
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init) - 1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                            initializer=float(param_init),
                                            trainable=agent.learn_penalty,
                                            dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty = tf.nn.softplus(penalty_param)  # swb:可以看作relu的平滑版本啊！！！(是一个实数啊！！)
    # swb: PPO-Lagrange的方法，这边的penalty的系数是可以进行学习的啊！！！！
    if agent.learn_penalty:
        if agent.penalty_param_loss:
            # swb: PPO-Lagrange的方法，这边的penalty_param_loss是True啊。。(机智啊！！！！)
            penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)  # swb:这个cur_cost_ph是一个placeholder啊
        else:
            penalty_loss = -penalty * (cur_cost_ph - cost_lim)
        train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)
        # swb:当agent总是违反约束时，penalty_param就会变得比较大，然后penalty也就变得很大了啊！！

    # =========================================================================#
    #  Create computation graph for policy learning                           #
    # =========================================================================#

    # Likelihood ratio
    ratio = tf.exp(logp - logp_old_ph)  # swb:ratio的shape也是（batch，）的向量啊

    # Surrogate advantage / clipped surrogate advantage
    if agent.clipped_adv:  # swb:tf.where的作用是根据第一个condition的结果，输出第二个，还是第三个。
        min_adv = tf.where(adv_ph > 0,
                           (1 + agent.clip_ratio) * adv_ph,
                           (1 - agent.clip_ratio) * adv_ph
                           )
        surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))  # swb：这句话说白了，就是拉高这些优势函数为正的action啊
    else:
        surr_adv = tf.reduce_mean(ratio * adv_ph)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cadv_ph)  # swb：这个是那个cost啊。。

    # Create policy objective function, including entropy regularization
    pi_objective = surr_adv + ent_reg * ent  # swb:这个是我们的原本ppo的更新目标，同时还加上了entropy啊！！！

    # Possibly include surr_cost in pi_objective
    if agent.objective_penalized:  # swb：这一步骤没有看明白啊！！！
        pi_objective -= penalty * surr_cost
        pi_objective /= (1 + penalty)  # swb:为什么这里要额外除以个 (1+penalty) 啊？！

    # Loss function for pi is negative of pi_objective
    pi_loss = -pi_objective
    # swb：这个pi_loss相当于，同时考虑了reward和cost啊 （a-λca）/(1+λ)
    # Optimizer-specific symbols
    if agent.trust_region:

        # Symbols needed for CG solver for any trust region method
        pi_params = get_vars('pi')  # swb:共轭梯度法要用到的东西啊！！(模型的参数啊！！！)
        flat_g = tro.flat_grad(pi_loss, pi_params)  # swb:这个是网络的梯度啊！！（被拉平了啊！！）
        v_ph, hvp = tro.hessian_vector_product(d_kl, pi_params)  # swb： v_ph是v的占位符，hvp是H*v的结果啊！！
        if agent.damping_coeff > 0:
            hvp += agent.damping_coeff * v_ph   # swb：请问这个damping_coef是用来干什么的啊！？

        # Symbols needed for CG solver for CPO only
        flat_b = tro.flat_grad(surr_cost, pi_params)  # swb:cost对于网络的求导数啊！！

        # Symbols for getting and setting params
        get_pi_params = tro.flat_concat(pi_params)  # swb:获得网络的参数啊！！
        set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)  # swb:获得

        training_package = dict(flat_g=flat_g,
                                flat_b=flat_b,
                                v_ph=v_ph,
                                hvp=hvp,
                                get_pi_params=get_pi_params,
                                set_pi_params=set_pi_params)

    elif agent.first_order:
        # Optimizer for first-order policy optimization
        # swb:相当于有两个优化器具啊！！！(前一个是对惩罚算子做优化，这一个是对policy的目标做优化啊。。)
        train_pi = MpiAdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)
        # Prepare training package for agent
        training_package = dict(train_pi=train_pi)
    else:
        raise NotImplementedError

    # Provide training package to agent
    # swb：training_package是一个字典，所以用字典添加成员啊。。
    training_package.update(dict(pi_loss=pi_loss,  # swb:train—package指的是训练所需要的东西啊！！
                                 surr_cost=surr_cost,
                                 d_kl=d_kl,
                                 target_kl=target_kl,
                                 cost_lim=cost_lim))
    agent.prepare_update(training_package)  # swb:这些东西是和agent训练有关系的，所以额外给单独里列了出来啊！！

    # =========================================================================#
    #  Create computation graph for value learning                            #
    # =========================================================================#

    # Value losses
    # swb：这里是v值的loss啊！！！！
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)
    vc_loss = tf.reduce_mean((cret_ph - vc) ** 2)

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    if agent.reward_penalized:
        total_value_loss = v_loss
    else:
        # swb:由于有两个网络分别评估reward和cost，所以这边需要把它们的loss加和在一起啊！！！
        total_value_loss = v_loss + vc_loss

    # Optimizer for value learning
    train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)

    # =========================================================================#
    #  Create session, sync across procs, and set up saver                    #
    # =========================================================================#

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())  # swb:这样，每个脚本之间的那个东西都是相同的啊！！

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v, 'vc': vc})
    # swb:这个东西会把网络的拓扑给保存下来啊！
    # summary_writer = tf.summary.FileWriter("log", sess.graph)
    # =========================================================================#
    #  Provide session to agent                                               #
    # =========================================================================#
    agent.prepare_session(sess)  # swb:为什么agent需要获得session呢？！

    # =========================================================================#
    #  Create function for running update (called at end of each epoch)       #
    # =========================================================================#
    # swb:每个epoch结束后，都会到这里进行学习啊！！！
    def update():  # swb:相当于会搜集30个episode左右的数据后，进入这边进行train和更新啊!!约30k的数据量
        cur_cost = logger.get_stats('EpCost')[0]  # swb:[0]是均值，[1]是std
        c = cur_cost - cost_lim  # swb:这个c就能知道，到底是违反了cost的约束，还是遵守了cost的约束。。
        if c > 0 and agent.cares_about_cost:  # swb:怪不得总是违反constraint啊，它老是报error的原因找到了啊。。
            logger.log('Warning! Safety constraint is already violated.', 'red')

        # =====================================================================#
        #  Prepare feed dict                                                   #
        # =====================================================================#
        # inputs_swb = {k: v for k, v in zip(buf_phs, buf.get())}  # swb:这样就能够把数据都给拿到了啊！！！(get过后，指针会清零啊！！)
        inputs = {k: v for k, v in zip(buf_phs, buf.get())}  # swb:这样就能够把数据都给拿到了啊！！！(get过后，指针会清零啊！！)

        # swb:我把下面注释掉了啊，好像下面这个没什么用处啊，所以我把它注释掉了啊！！
        # inputs[surr_cost_rescale_ph] = logger.get_stats('EpLen')[0]  # swb:这个就是那个平均的episode的长度啊！！！！
        inputs[cur_cost_ph] = cur_cost  # swb:平均每个episode，总共的cost数值！！！！

        # =====================================================================#
        #  Make some measurements before updating                             #
        # =====================================================================#

        measures = dict(LossPi=pi_loss,
                        SurrCost=surr_cost,
                        LossV=v_loss,
                        Entropy=ent)
        if not (agent.reward_penalized):
            measures['LossVC'] = vc_loss  # swb:PPO-Lagrangian用的就是这个！！
        if agent.use_penalty:
            measures['Penalty'] = penalty
        # swb:在更新之前的那个各种measures啊！！
        pre_update_measures = sess.run(measures, feed_dict=inputs)
        logger.store(**pre_update_measures)  # swb:这样就能把数值保存起来了啊！！(这个相当于每个epoch都要保存一次啊！！)
        # swb:问题，penalty是怎么对policy的更新起到重要的影响的呢？！
        # =====================================================================#
        #  Update penalty if learning penalty                                 #
        # =====================================================================#
        if agent.learn_penalty:  # swb:这个是去优化我们的penalty,大于limit后乘子变大，小于limit后乘子变小！！
            sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})  # swb:这个就是那个乘子啊！
        # swb:放在优化policy前面很重要啊，因为这相当于对这个policy的一个评估啊！！
        # =====================================================================#
        #  Update policy                                                      #
        # =====================================================================#
        agent.update_pi(inputs)

        # =====================================================================#
        #  Update value function                                              #
        # =====================================================================#
        for _ in range(vf_iters):  # swb：这里训练网络的时候，会同时优化value和cost两个网络啊
            sess.run(train_vf, feed_dict=inputs)  # swb:这个就是那个V值网络啊！

        # =====================================================================#
        #  Make some measurements after updating                              #
        # =====================================================================#

        del measures['Entropy']
        measures['KL'] = d_kl  # swb：再新测一次散度啊！！

        post_update_measures = sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:  # swb:假如这个项目是之前所拥有的那个！
                deltas['Delta' + k] = post_update_measures[k] - pre_update_measures[k]  # swb:意思是说，看看这些值的变化量罢了！！
        logger.store(KL=post_update_measures['KL'], **deltas)  # swb:记录更新前后的变化量，和policy的kl散度来衡量policy的变化程度

    # =========================================================================#
    #  Run main environment interaction loop                                  #
    # =========================================================================#

    start_time = time.time()
    o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    cur_penalty = 0
    cum_cost = 0  # swb:算的是总共的cost啊,一旦运行就不清零了啊！！

    for epoch in range(epochs):

        if agent.use_penalty:
            cur_penalty = sess.run(penalty)  # swb：手里拿到了惩罚系数(它直接就是一个Variable，所以可以直接拿到的啊！！)
        # swb:一定要跑完这么多个time-step才能进行更新啊！！！！！要不然的话，没有意义的啊！！！
        # swb:（因为你中间，可能环境没有跑到这么多个步数就中途停止了，所以要做各种各样的处理啊！！！）
        collect_start = time.time()
        for t in range(local_steps_per_epoch):
            # Possibly render
            if render and proc_id() == 0 and t < 1000:
                env.render()

            # Get outputs from policy
            get_action_outs = sess.run(get_action_ops,
                                       feed_dict={x_ph: o[np.newaxis]})
            a = get_action_outs['pi']
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']

            # Step in environment
            # swb:observation, reward, done, information:{cost, cost_hazards} (cost和cost_hazards有什么区别呢？？)
            o2, r, d, info = env.step(a)

            # Include penalty on cost
            c = info.get('cost', 0)

            # Track cumulative cost over training
            cum_cost += c

            # save and log
            if agent.reward_penalized:
                r_total = r - cur_penalty * c
                r_total = r_total / (1 + cur_penalty)
                buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
            else:  # swb:Larangian method 走的是这条通路啊！！！！
                buf.store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t)
            logger.store(VVals=v_t, CostVVals=vc_t)  # swb:epoch_dict保存了v和vc值啊!(每个time-step都会保存啊)

            o = o2  # swb: o_{t} = o_{t+1}
            ep_ret += r  # swb:一个回合的收益
            ep_cost += c  # swb：一个回合的损失
            ep_len += 1  # swb：一个回合的长度

            terminal = (d or (ep_len == max_ep_len))  # swb:要么done=True，或者episode-length等于1000了
            if terminal or (t == local_steps_per_epoch - 1):  # swb:要么就结束了，要么就这个epoch结束了。
                # swb：环境结束了，或者说到了最大的时间步长了！！
                # If trajectory didn't reach terminal state, bootstrap value target(s)
                if d and not (ep_len == max_ep_len):
                    # swb：这个判定条件意思是：环境强行完结，但是还没到episode的最大长度，就结束了。（可能中途有错，强行截断了啊）
                    # Note: we do not count env time out as true terminal state
                    last_val, last_cval = 0, 0  # swb：还没到最大回合长度就结束了，所以最后的value和cost都是0啊。。
                else:
                    feed_dict = {x_ph: o[np.newaxis]}  # swb：注意，这边是最新的观测啊！！
                    if agent.reward_penalized:
                        last_val = sess.run(v, feed_dict=feed_dict)
                        last_cval = 0
                    else:
                        last_val, last_cval = sess.run([v, vc], feed_dict=feed_dict)
                # swb:这边去求一些gae啊有关的东西！！(反正，一个episode结束之后，计算次GAE啊)
                buf.finish_path(last_val, last_cval)
                # Only save EpRet / EpLen if trajectory finished
                if terminal:  # swb：(相当于只保存每个epoch)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)  # swb:这三个值只是用来保存后进行观测啊，并不是说要拿来计算啊!!
                else:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)  # swb:这个就是说，环境还没run完，就被关掉了啊！！
                # swb:之后，我们就重置环境吧
                # Reset environment
                o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
        collect_end = time.time()  # swb:一次经验收集完毕啊
        msg(f"Collecting experience expands {collect_end-collect_start} seconds.")
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)  # swb：这个代码里面会同时保存环境和模型啊！！(为什么不是更新了后再保存啊？虽然差不多)

        # =====================================================================#
        #  Run RL update                                                      #
        # =====================================================================#
        update_start = time.time()  # swb:一次更新开始啊
        update()
        update_end = time.time()  # swb:一次更新完毕啊
        msg(f"Updating networks expands {update_end - update_start} seconds.")
        # =====================================================================#
        #  Cumulative cost calculations                                       #
        # =====================================================================#
        cumulative_cost = mpi_sum(cum_cost)  # swb:这个TM是一直累加的啊！！
        cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)  # swb：这个的平均值还会考虑训练初期的时候的结果啊！！

        # =====================================================================#
        #  Log performance and stats                                          #
        # =====================================================================#
        # swb:如果给了epoch， 那么会直接采用这个数据啊，否则就在epoch-buffer里面取啊
        logger.log_tabular('Epoch', epoch)
        # swb:with_min_and_max=True说明还需要额外保存最大值和最小值这两个东西。
        # Performance stats  swb：默认是保存std的，所以average_only=True的话只保存均值。with_min_and_max默认为False的啊！！
        logger.log_tabular('EpRet', with_min_and_max=True, tf_board=True)  # swb：一个epoch所有的 reward 啊！！ ep_ret += r
        logger.log_tabular('EpCost', with_min_and_max=True, tf_board=True)  # swb:一个epoch所有的 cost 啊！！ ep_cost += c
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Pi loss and change
        logger.log_tabular('LossPi', average_only=True, tf_board=True)
        logger.log_tabular('DeltaLossPi', average_only=True, tf_board=True)  # swb：这个应该是负数才是正常的啊！！

        # Surr cost and change
        logger.log_tabular('SurrCost', average_only=True, tf_board=True)
        logger.log_tabular('DeltaSurrCost', average_only=True, tf_board=True)  # swb：这个应该是负数才是正常的啊！！

        # V loss and change
        logger.log_tabular('LossV', average_only=True, tf_board=True)
        logger.log_tabular('DeltaLossV', average_only=True, tf_board=True)  # swb：这个应该是负数才是正常的啊！！

        # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        if not (agent.reward_penalized):
            logger.log_tabular('LossVC', average_only=True, tf_board=True)
            logger.log_tabular('DeltaLossVC', average_only=True, tf_board=True)  # swb：这个应该是负数才是正常的啊！！

        if agent.use_penalty or agent.save_penalty:
            logger.log_tabular('Penalty', average_only=True, tf_board=True)
            logger.log_tabular('DeltaPenalty', average_only=True, tf_board=True)
        else:
            logger.log_tabular('Penalty', 0)
            logger.log_tabular('DeltaPenalty', 0)

        # Anything from the agent?
        agent.log()

        # Policy stats
        logger.log_tabular('Entropy', average_only=True, tf_board=True)
        logger.log_tabular('KL', average_only=True, tf_board=True)

        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('Time', time.time() - start_time)

        # Show results!
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--len', type=int, default=1000)
    parser.add_argument('--cost_lim', type=float, default=10)
    parser.add_argument('--exp_name', type=str, default='runagent')
    parser.add_argument('--kl', type=float, default=0.01)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_penalized', action='store_true')
    parser.add_argument('--objective_penalized', action='store_true')
    parser.add_argument('--learn_penalty', action='store_true')
    parser.add_argument('--penalty_param_loss', action='store_true')
    parser.add_argument('--entreg', type=float, default=0.)
    args = parser.parse_args()
    print("You are stupid!")



