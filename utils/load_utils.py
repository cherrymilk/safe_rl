#!/usr/bin/env python

import joblib
import os
import os.path as osp
import tensorflow as tf
from utils.logx import restore_tf_graph


# swb：这边是在加载策略啊！！
def load_policy(fpath, itr='last', deterministic=False):
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    # load the things!
    sess = tf.Session(graph=tf.Graph())
    # swb：下面这边的画，sess这个对象会加载模型，然后又返回一个model（含有对应的placeholder和tensor啊）
    # swb：'x'也就是输入是placeholder，然后'v'，'vc'和'pi'都是tensor啊。所以需要喂进去的就是'x'啊！！
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']  # swb:如果想要实现确定性策略（也就是直接输出均值），就选择这个啊。
    else:
        print('Using default action op.')
        action_op = model['pi']  # swb：如果想要实现随机性策略（也就是要经过采样），就选择这个啊。

    # make function for producing an action given a single state (swb:我只能说，这个方法好机智啊）
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        # swb：所以说，这个joblib是怎么用的啊？！
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, sess


