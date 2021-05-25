#coding:utf-8
import textwrap
import argparse

# def parse_elapse_time(log_file):
#     """
#     Parse log file and calculate the elapse training time.
#     """
#     dy_time, st_time = [], []
#     with open(log_file,'r') as f:
#         for line in f:
#             content = line.strip('\n')
#             if 'ToStatic' not in content: continue
#             time = content.split(',')[-1]
#             assert 'ms' in time
#             time = float(time.split('=')[-1].strip())
#             if 'False' in content: 
#                 dy_time.append(time)
#             else:
#                 st_time.append(time)
    
#     assert len(dy_time) == len(st_time)
#     # drop warmup data
#     dy_time = dy_time[2:-2]
#     st_time = st_time[2:-2]

#     return mean(dy_time), mean(st_time)

# def parse_baseline_time(log_file):
#     """
#     Parse static program baseline training elapse time.
#     """
#     baseline_time = []
#     with open(log_file,'r') as f:
#         for line in f:
#             content = line.strip('\n')
#             if 'Baseline' not in content: continue
#             time = content.split(',')[-1]
#             assert 'ms' in time
#             time = float(time.split('=')[-1].strip())
#             baseline_time.append(time)
#     # drop warmup data
#     baseline_time = baseline_time[2:-2]

#     return mean(baseline_time)

def parse_throughout(log_file, baseline=False):
    dy_ips, st_ips, baseline_ips = [], [], []
    with open(log_file,'r') as f:
        for line in f:
            content = line.strip('\n')
            if 'ips' not in content: continue
            ips = float(content.split()[-2])
            if baseline:
                baseline_ips.append(ips)
            elif 'False' in content: 
                dy_ips.append(ips)
            else:
                st_ips.append(ips)
    if not baseline:
        assert len(dy_ips) == len(st_ips)
        # drop warmup data
        dy_ips = dy_ips[2:-1]
        st_ips = st_ips[2:-1]

        return mean(dy_ips), mean(st_ips)
    else:
        return mean(baseline_ips[1:-1])
    

def mean(nums):
    return sum(nums) / len(nums)


def to_mark_down(file, dy_ips, st_ips, baseline_ips, model_name, with_header=True, is_bert=False):
    header = textwrap.dedent("""
    | 训练吞吐 (img\|seq/s) | 静态图 |  动态图  |to_static(PE) | 提速(vs.动态图) | 提速(vs.静态图) | 
    |:------:|:------:|:------:|:------:|:------:|:------:|
    """)

    content = textwrap.dedent("""
    | {model}  | {baseline} | {dy}  |   {st} | {speedup}% | {gap}% | 
    """)
    speed_up = (st_ips - dy_ips) / dy_ips * 100
    gap = (st_ips - baseline_ips) / baseline_ips * 100
    content = content.format(
            model=model_name,
            baseline="%.2f" % baseline_ips, 
            dy="%.2f" % dy_ips, 
            st="%.2f" % st_ips,
            speedup= "%.2f" % speed_up,
            gap="%.2f" % gap,
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
        # ('mnist', './mnist_dy2static/mnist.log'),
        ('ptb_lm', './ptb_lm_dy2static/ptb_lm.log'),
        ('bert-base', 'bert_base_dy2static/bert_base.log'),
        # ('reinforcement_learning', './reinforcement_learning_dy2static/reinforcement_learning.log')
        ('mobilenet_v1', './mobile_net_dy2static/mobile_net_v1.log'),
        ('mobilenet_v2', './mobile_net_dy2static/mobile_net_v2.log'),
        # './sentiment_dy2static/sentiment.log',
        ('seresnet', './seresnet_dy2static/seresnet.log'),
        ('resnet', './resnet_dy2static/resnet.log'),
        # ('yolov3', 'yolov3_dy2static/yolov3.log'),
    ]

    return log_files


if __name__ == '__main__':
    args = parse_args()

    with_header = True
    for (model_name, log_file) in get_log_files():
        print("start to deal {}".format(log_file))
        bl_log_file = log_file[:-4] + "_static_baseline.log"
        dy_ips, st_ips = parse_throughout(log_file)
        baseline_ips = parse_throughout(bl_log_file, True)
        to_mark_down(args.output, dy_ips, st_ips, baseline_ips, model_name, with_header=with_header)
        with_header = False