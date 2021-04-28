#coding:utf-8
import textwrap
import argparse

def parse_elapse_time(log_file):
    dy_time, st_time = [], []
    with open(log_file,'r') as f:
        for line in f:
            content = line.strip('\n')
            if 'ToStatic' not in content: continue
            time = content.split(',')[-1]
            assert 'ms' in time
            time = float(time.split('=')[-1].strip())
            if 'False' in content: 
                dy_time.append(time)
            else:
                st_time.append(time)
    
    assert len(dy_time) == len(st_time)
    # drop warmup data
    dy_time = dy_time[5:-2]
    st_time = st_time[5:-2]

    return mean(dy_time), mean(st_time)

def mean(nums):
    return sum(nums) / len(nums)


def to_mark_down(file, dy_time, st_time, model_name, with_header=True):
    header = textwrap.dedent("""
    | 训练耗时 (ms/batch) |  动态图  | to_static(PE) | 提速 |
    |:------:|:------:|:------:|:------:|
    """)

    content = textwrap.dedent("""
    | {model}  |  {dy}  |   {st} | {speedup}% |
    """)

    speed_up = (dy_time - st_time) / st_time * 100
    content = content.format(
            model=model_name,
            dy="%.2f" % dy_time, 
            st="%.2f" % st_time,
            speedup= "%.2f" % speed_up
            ).lstrip('\n')

    with open(file, 'a') as f:
        if with_header:
            f.write(header)
        f.write(content)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default='replace_pe.md',
        type=str,
        required=True,
    )
    return parser.parse_args()

def get_log_files():
    log_files = [
        ('mnist', './mnist_dy2static/mnist.log'),
        ('ptb_lm', './ptb_lm_dy2static/ptb_lm.log'),
        ('resnet', './resnet_dy2static/resnet.log'),
        # ('reinforcement_learning', './reinforcement_learning_dy2static/reinforcement_learning.log')
        ('mobilenet_v1', './mobile_net_dy2static/mobile_net_v1.log'),
        ('mobilenet_v2', './mobile_net_dy2static/mobile_net_v2.log'),
        # # './sentiment_dy2static/sentiment.log',
        ('seresnet', './seresnet_dy2static/seresnet.log'),
        # ('yolov3', 'yolov3_dy2static/yolov3.log'),
    ]

    return log_files


if __name__ == '__main__':
    args = parse_args()

    with_header = True
    for (model_name, log_file) in get_log_files():
        print("start to deal {}".format(log_file))
        dy_time, st_time = parse_elapse_time(log_file)
        to_mark_down(args.output, dy_time, st_time, model_name, with_header=with_header)
        with_header = False