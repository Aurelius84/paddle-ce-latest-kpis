import paddle
import time
import numpy as np

from model import MobileNetV1, MobileNetV2, parse_args,print_arguments, reader_decorator,SEED

def optimizer_setting(args):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=args.base_lr,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))

    return optimizer

def train(args):
    # set random seed
    paddle.enable_static()
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    
    place = paddle.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda(
    ) and args.device == 'GPU' else paddle.CPUPlace()

    if "v1" in args.model_type:
        model = MobileNetV1(class_dim=args.class_num, scale=1.0)
    elif "v2" in args.model_type:
        model = MobileNetV2(class_dim=args.class_num, scale=1.0)
    else:
        print("wrong model name, please try model = v1 or v2")
        exit()

    dshape = [None, 3, 224, 224]

    input = paddle.static.data(name='data', shape=dshape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    predict = model(input)
    softmax_out = paddle.fluid.layers.softmax(predict, use_cudnn=False)
    cost = paddle.nn.functional.cross_entropy(input=softmax_out, label=label)
    avg_cost = paddle.mean(x=cost)

    acc_top1 = paddle.static.accuracy(
        input=predict, label=label, k=1)
    acc_top5 = paddle.static.accuracy(
        input=predict, label=label, k=5)

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

    for pass_id in range(args.pass_num):
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        cost_time = 0.

        for batch_id, data in enumerate(data_loader()):
            start_time = time.time()

            loss, acc1, acc5 = exe.run(
                compiled_prog,
                feed=data,
                fetch_list=[avg_cost, acc_top1, acc_top5])

            # cost time
            end_time = time.time()
            cost_time += (end_time - start_time) * 1000  # ms
            # append data of core indicators
            total_loss += loss[0]
            total_acc1 += acc1[0]
            total_acc5 += acc5[0]
            total_sample += 1

            if batch_id % args.log_internal == 0:
                ips = args.batch_size * args.log_internal / cost_time * 1000
                print( "StaticBaseline: \tPass = {},\tIter = {},\tLoss = {:.3f},\tAcc1 = {:.3f},\tAcc5 = {:.3f},\tElapse(ms) = {:.3f},\tips: {:.3f} img/s\n".format
                    ( pass_id, batch_id, total_loss / total_sample, \
                        total_acc1 / total_sample, total_acc5 / total_sample, cost_time / args.log_internal, ips))
                # reset cost_time
                cost_time = 0.
            if batch_id / args.log_internal > 10:
                break


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train(args)