import paddle
import time
import numpy as np

from model import MNIST, parse_args, print_arguments, SEED, AdamOptimizer

def train(model, args):
    place = paddle.CUDAPlace(0)
    # set random seed to initialize parameters
    paddle.static.default_main_program().random_seed = SEED
    paddle.static.default_startup_program().random_seed = SEED

    dshape = [None, 28, 28]

    input = paddle.static.data(name='data', shape=dshape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    prediction, acc, avg_cost  = model(input, label)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True

    optimizer = AdamOptimizer(learning_rate=0.001)
    opts = optimizer.minimize(avg_cost)

    # load flowers data
    train_dataset = paddle.vision.datasets.MNIST(mode='train', backend='cv2')
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    compiled_prog = paddle.static.CompiledProgram(paddle.static.default_main_program()).with_data_parallel(
        loss_name=avg_cost.name, build_strategy=build_strategy)

    for pass_id in range(args.pass_num):
        cost_time = []

        for batch_id, data in enumerate(train_loader()):
            start_time = time.time()
            pred, acc1, loss = exe.run(
                compiled_prog,
                feed={'data': data[0], 'label': data[1]},
                fetch_list=[prediction, acc, avg_cost])
            
            end_time = time.time()
            cost_t = (end_time - start_time) * 1000  # ms
            cost_time.append(cost_t)

            if batch_id % args.log_internal == 0:
                print(
                    "StaticBaseline: Pass = %d, Iter = %d, Loss = %.3f, Accuracy = %.3f, Elapse(ms) = %.3f, ips: %.3f img/s\n"
                    % (pass_id, batch_id, loss[0], acc1[0], np.mean(cost_time), args.batch_size/np.mean(cost_time) * 1000))
                cost_time = []

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    paddle.enable_static()
    model = MNIST()
    train(model, args)