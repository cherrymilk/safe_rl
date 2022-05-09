from tensorboardX import SummaryWriter
import os.path as osp
import pandas as pd


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


# swb:在命令行中，打印不同颜色的文字啊！！
def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class EpochLogger:
    def __init__(self, output_dir, file_name="progress.csv"):
        self.epoch_dict = dict()
        assert output_dir is not None
        self.output_dir = output_dir
        self.tf_board_logger = SummaryWriter(output_dir)
        self.out_file_path = osp.join(output_dir, file_name)
        self.log(f"Logging data to {self.out_file_path}")

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def store(self, **kwargs):
        # swb:把数据保存到我们的epoch-dict里面去
        for k, v in kwargs.items():
            # swb：假如没有的花，那么我们就创建啊！！
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def tf_board_plot(self, key, value, iterations, category='train_log', save=True):
        self.tf_board_logger.add_scalar(f"{category}/{key}", value, iterations)
        if save is True:
            self.store(key=value)

    def dump_2_file(self):
        data = pd.DataFrame(self.epoch_dict)
        data.to_csv(self.out_file_path)


if __name__ == "__main__":
    logger = EpochLogger("./logger/test")
    key = "utils"
    value = 32
    logger.store(key=value)
    logger.store(wow=12)
    logger.store(key=value+1)
    logger.store(wow=13)
    logger.store(key=value+2)
    logger.store(wow=14)
    logger.store(key=value+3)
    logger.store(wow=15)
    logger.dump_2_file()
    print("finished!!")








