import paddle
import time
import math
import numpy as np

from model import PtbModel, parse_args, print_arguments, SEED, getdata

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
def train(args):
    # set random seed
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    place = paddle.CUDAPlace(0)

    # create model
    num_layers = 1
    hidden_size = 10
    num_steps = 30
    init_scale = 0.1
    dropout = 0.0
    vocab_size = 10000
    model = PtbModel(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_steps=num_steps,
        init_scale=init_scale,
        dropout=dropout)

    x = paddle.static.data(name='x', shape=[None, num_steps, 1], dtype='int64')
    y = paddle.static.data(name='y', shape=[None, num_steps, 1], dtype='int64')
    init_hidden = paddle.zeros(
            (num_layers, args.batch_size, hidden_size), dtype='float32')
    init_hidden.stop_gradient = True
    init_cell = paddle.zeros(
        (num_layers, args.batch_size, hidden_size), dtype='float32')
    init_cell.stop_gradient=True

    avg_cost, last_hidden, last_cell = model(x, y, init_hidden, init_cell)
    optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
    opts = optimizer.minimize(avg_cost)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True

    # load data
    print("begin to load data")
    ptb_data = getdata.get_ptb_data(args.batch_size, num_steps)
    print("finished load data")

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    compiled_prog = paddle.static.CompiledProgram(paddle.static.default_main_program()).with_data_parallel(
        loss_name=avg_cost.name, build_strategy=build_strategy)

    for epoch_id in range(args.pass_num):
        total_loss = 0.0
        total_sample = 0
        cost_time = 0.

        for step_id, data in enumerate(ptb_data):
            start_time = time.time()
            x_data = data[0]
            y_data = data[1]

            x_data = x_data.reshape((-1, num_steps, 1))
            y_data = y_data.reshape((-1, num_steps, 1))

            loss, _, _ = exe.run(
                compiled_prog,
                feed={'x': x_data, 'y': y_data},
                fetch_list=[avg_cost, last_hidden, last_cell])
                  
            # cost time
            end_time = time.time()
            cost_time += (end_time - start_time) * 1000  # ms
            # append data of core indicators
            total_loss += loss[0]
            total_sample += 1

            # print log
            if step_id % args.log_internal == 0:
                print(
                    'Static Baseline: pass = %d, Iter %d, Loss = %0.3f, Elapse(ms) = %.3f\n'
                    % (epoch_id, step_id, loss, cost_time / args.log_internal))
                # reset cost_time
                cost_time = 0.
            if step_id == 300:
                break

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    paddle.enable_static()
    train(args)