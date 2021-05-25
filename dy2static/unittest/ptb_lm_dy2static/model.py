# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import time
import argparse
import os
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.jit import ProgramTranslator

import getdata

SEED = 2020


def parse_args():
    parser = argparse.ArgumentParser("ptb_lm model benchmark.")
    # parser.add_argument(
    #     '--to_static', type=bool, default=True, help='whether to train model in static mode.')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--pass_num', type=int, default=5, help='The number of passes.')
    parser.add_argument(
        '--log_internal',
        type=int,
        default=20,
        help='The internal step of log.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


class SimpleLSTMRNN(paddle.nn.Layer):
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Uniform(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        cell_array = []
        hidden_array = []

        for i in range(self._num_layers):
            hidden_array.append(init_hidden[i])
            cell_array.append(init_cell[i])

        res = []
        for index in range(self._num_steps):
            step_input = input_embedding[:, index, :]
            for k in range(self._num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = paddle.concat(x=[step_input, pre_hidden], axis=1)
                gate_input = paddle.matmul(x=nn, y=weight_1)

                gate_input = paddle.add(x=gate_input, y=bias)
                i, j, f, o = paddle.split(
                    x=gate_input, num_or_sections=4, axis=-1)
                c = pre_cell * paddle.nn.functional.sigmoid(
                    f) + paddle.nn.functional.sigmoid(i) * paddle.tanh(j)
                m = paddle.tanh(c) * paddle.nn.functional.sigmoid(o)
                hidden_array[k] = m
                cell_array[k] = c
                step_input = m

                if self._dropout is not None and self._dropout > 0.0:
                    step_input = paddle.nn.functional.dropout(
                        step_input, p=self._dropout, mode='upscale_in_train')
            res.append(step_input)
        real_res = paddle.concat(x=res, axis=1)
        real_res = paddle.reshape(real_res,
                                  [-1, self._num_steps, self._hidden_size])
        last_hidden = paddle.concat(x=hidden_array, axis=1)
        last_hidden = paddle.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = paddle.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = paddle.concat(x=cell_array, axis=1)
        last_cell = paddle.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = paddle.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(paddle.nn.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 dropout=None):
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = paddle.fluid.dygraph.nn.Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=paddle.ParamAttr(
                #name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale))

    def build_once(self, input, label, init_hidden, init_cell):
        pass

    def forward(self, input, label, init_hidden, init_cell):
        init_h = paddle.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = paddle.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])
        x_emb = self.embedding(input)

        x_emb = paddle.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = paddle.nn.functional.dropout(
                x_emb, p=self.dropout, mode='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)
        projection = paddle.matmul(x=rnn_out, y=self.softmax_weight)
        projection = paddle.add(x=projection, y=self.softmax_bias)

        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = paddle.reshape(loss, shape=[-1, self.num_steps])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.fluid.layers.reduce_sum(loss)

        return loss, last_hidden, last_cell

    def debug_emb(self):

        np.save("emb_grad", self.x_emb.gradient())


def train(args, to_static=False):
    # whether to apply dy2stat
    prog_trans = ProgramTranslator()
    prog_trans.enable(to_static)
    # set device
    device = 'gpu:0' if fluid.is_compiled_with_cuda(
    ) and args.device == 'GPU' else 'cpu'
    paddle.set_device(device)
    # set random seed to initialize parameters
    fluid.default_main_program().random_seed = SEED
    fluid.default_startup_program().random_seed = SEED

    # create model
    num_layers = 1
    hidden_size = 10
    num_steps = 30
    init_scale = 0.1
    dropout = 0.0
    vocab_size = 10000
    ptb = PtbModel(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_steps=num_steps,
        init_scale=init_scale,
        dropout=dropout)
    if to_static:
        ptb = paddle.jit.to_static(ptb)
    sgd = paddle.optimizer.SGD(learning_rate=1e-3, parameters=ptb.parameters())

    # load data
    print("begin to load data")
    ptb_data = getdata.get_ptb_data(args.batch_size, num_steps)
    print("finished load data")

    place = paddle.CUDAPlace(0)

    for pass_id in range(args.pass_num):
        # core indicators
        cost_time = 0.
        loss = []
        init_hidden_data = np.zeros(
            (num_layers, args.batch_size, hidden_size), dtype='float32')
        init_cell_data = np.zeros(
            (num_layers, args.batch_size, hidden_size), dtype='float32')

        init_hidden = paddle.to_tensor(
            data=init_hidden_data, place=place, stop_gradient=True)
        init_cell = paddle.to_tensor(
            data=init_cell_data, place=place, stop_gradient=True)
        for batch_id, data in enumerate(ptb_data):
            batch_start = time.time()
            x_data = data[0]
            y_data = data[1]

            x_data = x_data.reshape((-1, num_steps, 1))
            y_data = y_data.reshape((-1, num_steps, 1))

            x = paddle.to_tensor(
                data=x_data, place=place, stop_gradient=True)
            y = paddle.to_tensor(
                data=y_data, place=place, stop_gradient=True)

            dy_loss, last_hidden, last_cell = ptb(x, y, init_hidden, init_cell)
            out_loss = dy_loss.numpy()
            dy_loss.backward()
            sgd.minimize(dy_loss)
            ptb.clear_gradients()

            batch_end = time.time()
            cost_t = (batch_end - batch_start) * 1000  # ms
            cost_time += cost_t
            loss.append(out_loss)

            if batch_id % args.log_internal == 0:
                ips = args.batch_size * args.log_internal / cost_time * 1000
                print(
                    'ToStatic = %s, pass = %d, Iter %d, Loss = %0.3f, Elapse(ms) = %f, ips = %0.3f seq/s\n'
                    % (to_static, pass_id, batch_id, out_loss, cost_time / args.log_internal, ips))
                cost_time = 0.
            if batch_id / args.log_internal > 15:
                break

    ret = out_loss, last_hidden.numpy(), last_cell.numpy()
    return ret


def run_benchmark(args):
    # train in dygraph mode
    print('dygraph mode')
    train(args, to_static=False)

    print('static mode')
    # train in static mode
    train(args, to_static=True)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    run_benchmark(args)
