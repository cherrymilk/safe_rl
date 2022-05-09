import numpy as np
import tensorflow as tf
from mpi4py import MPI
from utils.mpi_tools import broadcast


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)  # swb:全部都转化为1纬的vector啊!


# swb:这个的骚操作就是将向量化的参数给赋值过去啊！！
def assign_params_from_flat(x, params):
    # swb:x是已经压扁过后的tensor，params是没有经过压扁的tensors们，所以可以根据params的信息恢复出信息啊！！
    flat_size = lambda p : int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])


def sync_params(params):
    get_params = flat_concat(params)  # swb:这边把所有的变量都拿到了，然后把它们都给压扁了，然后给串儿起来了啊！！
    def _broadcast(x):
        broadcast(x)  # swb:这边经过了广播后，x都是一样的了，全部向0号线程看齐。。
        return x
    synced_params = tf.py_func(_broadcast, [get_params], tf.float32)  # swb:我知道了，因为这个操作不是tf提供的，所以需要拿py_func包装下
    return assign_params_from_flat(synced_params, params)  # swb：大家的synced_params都是一样的了，然后就要把它们赋值到模型上去


def sync_all_params():
    """Sync all tf variables across MPI processes."""
    return sync_params(tf.global_variables())  # swb:这个到底是怎么同步的啊！？


class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """
    Adam optimizer that averages gradients across MPI processes.

    The compute_gradients method is taken from Baselines `MpiAdamOptimizer`_. 
    For documentation on method arguments, see the Tensorflow docs page for 
    the base `AdamOptimizer`_.

    .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py
    .. _`AdamOptimizer`: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    """
    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)  # swb:调用父类的求梯度的函数啊
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]
        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])



