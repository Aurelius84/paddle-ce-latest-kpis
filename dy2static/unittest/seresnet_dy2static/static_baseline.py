import paddle
import time
import math
import numpy as np

from model import SeResNeXt, parse_args,print_arguments, SEED, reader_decorator

def optimizer_setting(args):
    params = {
        "learning_strategy": {
            "name": "cosine_decay",
            "batch_size": args.batch_size,
            "epochs": [40, 80, 100],
            "steps": [0.1, 0.01, 0.001, 0.0001]
        },
        "lr": args.base_lr,
        "total_images": 6149,
        "momentum_rate": args.momentum_rate,
        "l2_decay": args.l2_decay,
        "num_epochs": args.pass_num,
    }
    ls = params["learning_strategy"]
    if "total_images" not in params:
        total_images = 6149
    else:
        total_images = params["total_images"]

    batch_size = ls["batch_size"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]

    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    optimizer = paddle.fluid.optimizer.Momentum(
        learning_rate=paddle.fluid.layers.cosine_decay(
            learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
        momentum=momentum_rate,
        regularization=paddle.fluid.regularizer.L2Decay(l2_decay))

    return optimizer
def train(model, args):
    # set random seed
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    place = paddle.CUDAPlace(0)

    dshape = [None, 3, 224, 224]
    input = paddle.static.data(name='data', shape=dshape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    out, avg_cost, acc_top1, acc_top5 = model(input, label)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True

    optimizer = optimizer_setting(args)
    opts = optimizer.minimize(avg_cost)

    # load flowers data
    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
        batch_size=args.batch_size,
        drop_last=True)
    data_loader = paddle.io.DataLoader.from_generator(
        capacity=5, iterable=True, feed_list=[input, label])
    data_loader.set_sample_list_generator(train_reader, places=[place])


    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    compiled_prog = paddle.static.CompiledProgram(paddle.static.default_main_program()).with_data_parallel(
        loss_name=avg_cost.name, build_strategy=build_strategy)

    for epoch_id in range(args.pass_num):
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        cost_time = 0.

        for step_id, data in enumerate(data_loader()):
            start_time = time.time()

            pred, loss, acc1, acc5 = exe.run(
                compiled_prog,
                feed=data,
                fetch_list=[out, avg_cost, acc_top1, acc_top5])
                  
            # cost time
            end_time = time.time()
            cost_time += (end_time - start_time) * 1000  # ms
            # append data of core indicators
            total_loss += loss[0]
            total_acc1 += acc1[0]
            total_acc5 += acc5[0]
            total_sample += 1

            # print log
            if step_id % args.log_internal == 0:
                ips = args.batch_size * args.log_internal / cost_time * 1000
                print( "StaticBaseline\tPass = {},\tIter = {},\tLoss = {:.3f},\tAcc1 = {:.3f},\tAcc5 = {:.3f},\tElapse(ms) = {:.3f},\tips = {:.3f} img/s\n".format
                    (epoch_id, step_id, total_loss / total_sample, \
                        total_acc1 / total_sample, total_acc5 / total_sample, cost_time / args.log_internal,ips))
                # reset cost_time
                cost_time = 0.
            if step_id / args.log_internal > 10:
                break

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    paddle.enable_static()
    model = SeResNeXt()
    train(model, args)