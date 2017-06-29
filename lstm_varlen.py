# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import cPickle as pickle
import numpy as np
import argparse

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import loss
from singa import device
from singa import tensor
from singa import optimizer
from singa import initializer
from singa.proto import model_pb2
from singa import utils

from data_loader import *
from time import time

def numpy2tensors(npx, dev):
    '''batch, seq, dim -- > seq, batch, dim'''
    #tmpx = np.swapaxes(npx, 0, 1)
    tmpx = npx
    inputs = []
    for t in range(tmpx.shape[0]):
        x = tensor.from_numpy(tmpx[t])
        x.to_device(dev)
        inputs.append(x)
    return inputs

def convert(data, batch_size, seq_length, vocab_size, dev):
    '''convert a batch of data into a sequence of input tensors'''
    x = np.zeros((batch_size, seq_length, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        for t in range(seq_length):
            c = data[b, t]
            x[b, t, c] = 1
    return x
    #return numpy2tensors(x, dev)

def convert_sample(data, seq_length, vocab_size, dev):
    '''convert a batch of data into a sequence of input tensors'''
    x = np.zeros((seq_length, vocab_size), dtype=np.float32)
    for t in range(seq_length):
        x[t, data[t]] = 1
    return x


def get_lr(epoch):
    return 0.01 / float(1 << (epoch / 50))


class SANet(object):
    def __init__(self, name, dev, num_stack_layers, embed_size,
            hidden_size, seq_length, vocab_size, batchsize=32):
        self.name = name
        self.dev = dev
        self.batchsize = batchsize
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_stack_layers = num_stack_layers

    def create_net(self, dropout=0.5):
        self.embed = layer.Dense('embed', 
                        self.embed_size,
			input_sample_shape=(self.vocab_size,
			))
        self.embed.to_device(self.dev)

        self.lstm = layer.LSTM(
                    name='lstm',
                    hidden_size=self.hidden_size,
                    num_stacks=self.num_stack_layers,
                    dropout=dropout,
                    input_sample_shape=(
                        self.embed_size,
                    ))
        self.lstm.to_device(self.dev)

        self.dense = layer.Dense(
                    'dense',
                    2, #output shape
                    input_sample_shape=(
                        self.hidden_size,
                    ))
        self.dense.to_device(self.dev)
        self.sft = layer.Softmax('softmax', 
	             input_sample_shape=(
                        2,
                    ))
        self.sft.to_device(self.dev)

        self.loss = loss.SoftmaxCrossEntropy()


    def train(self, data_path, max_epoch, model_path='model'):
        # SGD with L2 gradient normalization
        opt = optimizer.RMSProp(constraint=optimizer.L2Constraint(5))
        #opt = optimizer.SGD(momentum=0.9, weight_decay=5e-4)

        # initialize embedding layer
        embed_w = self.embed.param_values()[0]
        embed_b = self.embed.param_values()[1]
        #initializer.uniform(embed_w, 0, embed_w.shape[1])
        embed_w.uniform(-0.08, 0.08)
        embed_b.set_value(0)
        print 'embed weight l1 = %f' % (embed_w.l1())
        print 'embed b l1 = %f' % (embed_b.l1())

        # initialize lstm layer
        lstm_w = self.lstm.param_values()[0]
        lstm_w.uniform(-0.08, 0.08)  # init all lstm parameters
        print 'lstm weight l1 = %f' % (lstm_w.l1())

        # initialize dense layer
        dense_w = self.dense.param_values()[0]
        dense_b = self.dense.param_values()[1]
        dense_w.uniform(-0.1, 0.1)
        dense_b.set_value(0)
        print 'dense w ', dense_w.shape
        print 'dense b ', dense_b.shape
        print 'dense weight l1 = %f' % (dense_w.l1())
        print 'dense b l1 = %f' % (dense_b.l1())

        start = time()
        train_dat, train_label, val_dat, val_label = load_sample() 
        #train_dat, train_label, val_dat, val_label = load_corpus(data_path)
        train_label = word2onehot(train_label, 2)
        val_label = word2onehot(val_label, 2)
        print 'loading time:', time() - start
        print "train data shape:", train_dat.shape, "train label shape:", train_label.shape
        print "val data shape:", val_dat.shape, "val label shape:", val_label.shape
        for epoch in range(max_epoch):
            train_loss = 0
            num_train_batch = train_dat.shape[0] / self.batchsize
            glb_acc = 0
            for b in range(num_train_batch):
                start = time()
                # load training data
                inputs_arr = train_dat[b * self.batchsize: (b + 1) * self.batchsize]
                labels = train_label[b * self.batchsize: (b + 1) * self.batchsize]
                lens = rm_padding(inputs_arr)
                acc = 0
                batch_loss = 0.0
                g_dense_w = tensor.Tensor(dense_w.shape, self.dev)
                g_dense_w.set_value(0)
                g_dense_b = tensor.Tensor(dense_b.shape, self.dev)
                g_dense_b.set_value(0)
                g_lstm_w = tensor.Tensor(lstm_w.shape, self.dev)
                g_lstm_w.set_value(0)
                g_embed_w = tensor.Tensor(embed_w.shape, self.dev)
                g_embed_w.set_value(0)
                for idx_sam in range(len(inputs_arr)):
                    sam_arr = inputs_arr[idx_sam]
                    sam_arr = convert_sample(sam_arr, sam_arr.shape[0], self.vocab_size, self.dev)
                    sample = tensor.from_numpy(sam_arr)
                    sample.to_device(self.dev)
                    #print sample.shape
                    embed = self.embed.forward(model_pb2.kTrain, sample)
                    #print embed.shape is (53, 128)
                    # embed.shape[0] means the sequence length of the sample
                    embeded = []
                    for idx_seq in range(self.seq_length):
                        if idx_seq >= embed.shape[0]:
                            embeded.append(tensor.Tensor())
                        else:
                            seq = tensor.Tensor((1,embed.shape[1]), self.dev)
                            tensor.copy_data_to_from(seq, embed, embed.shape[1], 0, idx_seq * embed.shape[1])
                            embeded.append(seq)
                    embeded.append(tensor.Tensor()) # hx
                    embeded.append(tensor.Tensor()) # cx
                    #print 'forward embedding time:', time() -start
                    #print tensor.to_numpy(embeded[self.seq_length-1])
                   
                    # forward lstm layer
                    hidden = self.lstm.forward(model_pb2.kTrain, embeded)
                    # outputs are [y1, ..., yn, hx, cx], only need the last output as the predicted latent vector
                    #print len(hidden), hidden[embed.shape[0]-1]
                    #print [hidden[i].l1() for i in range(len(hidden))]
                    # forward dense and loss layer
                    act = self.dense.forward(model_pb2.kTrain, hidden[lens[idx_sam]-1])
                    label = tensor.from_numpy(labels[idx_sam])
                    label.to_device(self.dev)
                    lvalue = self.loss.forward(model_pb2.kTrain, act, label)
                    #print 'forward dense time:', time() - start
                    regularized_act = self.sft.forward(model_pb2.kEval, act)
                    pred = tensor.to_numpy(regularized_act)
                    gt = labels[idx_sam][1]
                    if (gt and pred[0,1] > pred[0,0]) or (gt == 0 and pred[0,1] <= pred[0,0]):
                        acc += 1
                
                    grads = []
                    batch_loss += lvalue.l1() / self.batchsize
                    #print batch_loss
                    start = time()
                    # backward loss and dense layer
                    grad = self.loss.backward() / self.batchsize
                    grad, gwb = self.dense.backward(model_pb2.kTrain, grad)
                    g_dense_w += gwb[0]
                    g_dense_b += gwb[1]
                    #print 'dense_w l1 = %f' % (gwb[0].l1())
                    for i in range(self.seq_length):
                        if i == lens[idx_sam] - 1:
                            grads.append(grad)
                        else:
                            emp = tensor.Tensor(grad.shape, self.dev)
                            emp.set_value(0)
                            grads.append(emp)
                    grads.append(tensor.Tensor())
                    grads.append(tensor.Tensor())
                    # backward lstm layer
                    lstm_input_grad, lstm_param_grad = self.lstm.backward(model_pb2.kTrain, grads)
                    g_lstm_w += lstm_param_grad[0] 
                    #print 'lstm_input l1 = %f' % (lstm_input_grad[0].l1())
                    #print 'backward lstm'  
                    embed_grad = tensor.Tensor(embed.shape, self.dev)
                    for idx in range(len(lstm_input_grad)-2):
                        tensor.copy_data_to_from(embed_grad, lstm_input_grad[idx], embed.shape[1],
					idx * embed.shape[1], 0)
                    _, grad_w = self.embed.backward(model_pb2.kTrain, embed_grad)
                    #print 'backward embedding time:', time() - start
                    #print 'embed weight l1 = %f' % (grad_w[0].l1())
                    g_embed_w += grad_w[0]

                train_loss += batch_loss
                glb_acc += acc

                utils.update_progress(
                    b * 1.0 / num_train_batch, 'training loss = %f, acc = %f' %
                    (batch_loss, acc * 1.0 / self.batchsize))
                opt.apply_with_lr(epoch, get_lr(epoch), g_lstm_w, lstm_w, 'lstm_w')
                opt.apply_with_lr(epoch, get_lr(epoch), g_dense_w, dense_w, 'dense_w')
                opt.apply_with_lr(epoch, get_lr(epoch), g_dense_b, dense_b, 'dense_b')
                opt.apply_with_lr(epoch, get_lr(epoch), g_embed_w, embed_w, 'embed_w')
                #opt.apply_with_lr(epoch, get_lr(epoch), grad_w[1], embed_b, 'embed_b')
            print '\nEpoch %d, train loss is %f, acc = %f' % \
                    (epoch, train_loss / num_train_batch, glb_acc * 1. / (self.batchsize * num_train_batch))

            # evaluation
            eval_loss = 0
            val_acc = 0
            num_test_batch = min(5000, val_dat.shape[0] / self.batchsize)
            for b in range(num_test_batch):
                acc = 0
                val_arr = val_dat[b * self.batchsize: (b + 1) * self.batchsize]
                labels = val_label[b * self.batchsize: (b + 1) * self.batchsize]
                lens = rm_padding(val_arr)
                val_arr = convert(val_arr, self.batchsize, self.seq_length,
                                  self.vocab_size, self.dev)
                val_arr = np.swapaxes(val_arr, 0, 1).reshape((
			self.batchsize * self.seq_length, self.vocab_size)) 
                inputs = tensor.from_numpy(val_arr)
                inputs.to_device(self.dev) # shape (128*53, 33366)
                embed = self.embed.forward(model_pb2.kEval, inputs)
                embed.reshape((self.seq_length, self.batchsize, self.embed_size))
                embeded = []
                for idx in range(self.seq_length):
                    embed_seq = tensor.Tensor((self.batchsize, self.embed_size), self.dev)
                    tensor.copy_data_to_from(embed_seq, embed, 
			self.batchsize * self.embed_size, 0, idx * self.batchsize * self.embed_size)
                    embeded.append(embed_seq)
                embeded.append(tensor.Tensor()) # hx
                embeded.append(tensor.Tensor()) # cx

                hidden = self.lstm.forward(model_pb2.kEval, embeded)
                hidden_batch = tensor.Tensor((self.batchsize, self.hidden_size), self.dev)
                for idx in range(self.batchsize):
                    tensor.copy_data_to_from(hidden_batch, hidden[lens[idx]-1],
			self.hidden_size, idx * self.hidden_size, idx* self.hidden_size)

                act = self.dense.forward(model_pb2.kEval, hidden_batch)
                labels = tensor.from_numpy(labels)
                labels.to_device(self.dev)
                eval_loss += self.loss.forward(model_pb2.kEval, act, labels).l1()
                regularized_act = self.sft.forward(model_pb2.kEval, act)
                pred = tensor.to_numpy(regularized_act)
                gt = tensor.to_numpy(labels)[:,1]
                for i in range(self.batchsize):
                    if (gt[i] and pred[i,1] > pred[i,0]) or (gt[i] == 0 and pred[i,1] <= pred[i,0]):
                        acc += 1
                #print 'acc = %f' % (acc * 1. / self.batchsize)
                val_acc += acc
  
            print 'Epoch %d, evaluation loss is %f, acc = %f' % \
                (epoch, eval_loss / num_test_batch, val_acc * 1. / (num_test_batch * self.batchsize))

            # model saving
            if (epoch + 1) % 2 == 0 or epoch + 1 == max_epoch:
                print 'dense weight l1 = %f' % (dense_w.l1())
                print 'dense bias l1 = %f' % (dense_b.l1())
                print 'lstm weight l1 = %f' % (lstm_w.l1())
                print 'embed weight l1 = %f' % (embed_w.l1())
                # checkpoint the file model
                with open('%s_%d.bin' % (model_path, epoch), 'wb') as fd:
                    print 'saving model to %s' % model_path
                    d = {}
                    for name, w in zip(
                        ['embed_w','embed_b', 'lstm_w', 'dense_w', 'dense_b'],
                        [embed_w, embed_b, lstm_w, dense_w, dense_b]):
                        w.to_host()
                        d[name] = tensor.to_numpy(w)
                        w.to_device(self.dev)
                    '''d['idx_to_char'] = data.idx_to_char
                    d['char_to_idx'] = data.char_to_idx
                    d['hidden_size'] = hidden_size
                    d['num_stacks'] = num_stacks
                    d['dropout'] = dropout'''
                    pickle.dump(d, fd)


    def load(self, model_path='model'):
        with open('%s.bin' % (model_path), 'rb') as fd:
            params = pickle.load(fd)

        # set embedding params
        embed_w = self.embed.param_values()[0]
        embed_b = self.embed.param_values()[1]
        embed_w.copy_from_numpy(params['embed_w'])
        embed_w.to_device(self.dev)
        embed_b.set_value(0)
        print 'embed weight l1 = %f' % (embed_w.l1())

        # set lstm params
        lstm_w = self.lstm.param_values()[0]
        lstm_w.copy_from_numpy(params['lstm_w'])
        lstm_w.to_device(self.dev)
        print 'lstm weight l1 = %f' % (lstm_w.l1())

        # set dense params
        dense_w = self.dense.param_values()[0]
        dense_b = self.dense.param_values()[1]
        print 'dense w ', dense_w.shape
        print 'dense b ', dense_b.shape
        dense_w.copy_from_numpy(params['dense_w'])
        dense_w.to_device(self.dev)
        print 'dense weight l1 = %f' % (dense_w.l1())
        dense_b.copy_from_numpy(params['dense_b'])
        dense_b.to_device(self.dev)
        print 'dense b l1 = %f' % (dense_b.l1())


    def inference(self, data, batchsize=1, model_path='model'):
        lens = rm_padding(data)
        input_arr = convert(data, batchsize, self.seq_length,
                          self.vocab_size, self.dev)
        input_arr = np.swapaxes(input_arr, 0, 1).reshape((
			batchsize * self.seq_length, self.vocab_size)) 
        inputs = tensor.from_numpy(input_arr)
        inputs.to_device(self.dev)
        embed = self.embed.forward(model_pb2.kEval, inputs)
        embeded = []
        for idx in range(self.seq_length):
            point = tensor.Tensor((batchsize, self.embed_size), self.dev)
            tensor.copy_data_to_from(point, embed, batchsize * self.embed_size,
			0, idx * batchsize * self.embed_size)
            embeded.append(point)
        embeded.append(tensor.Tensor()) # hx
        embeded.append(tensor.Tensor()) # cx

        hidden = self.lstm.forward(model_pb2.kEval, embeded)
        hidden_batch = tensor.Tensor((batchsize, self.hidden_size), self.dev)
        for idx in range(batchsize):
            tensor.copy_data_to_from(hidden_batch, hidden[lens[idx]-1],
		self.hidden_size, idx * self.hidden_size, idx* self.hidden_size)

        act = self.dense.forward(model_pb2.kEval, hidden_batch)
        probs = self.sft.forward(model_pb2.kEval, act)
        probs = tensor.to_numpy(probs)
        return probs[:,1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-stack LSTM for '
        'modeling  character sequence from plain text files')
    parser.add_argument('data', type=str, help='training file')
    parser.add_argument('-b', type=int, default=128, help='batch_size')
    parser.add_argument('-l', type=int, default=53, help='sequence length')
    parser.add_argument('-e', type=int, default=128, help='embed size')
    parser.add_argument('-d', type=int, default=100, help='hidden size')
    parser.add_argument('-s', type=int, default=1, help='num of stacks')
    parser.add_argument('-m', type=int, default=10, help='max num of epoch')
    parser.add_argument('-v', type=int, default=33366, help='vocabulary size')
    parser.add_argument('-f', type=int, default=True, help='prediction flag')
    args = parser.parse_args()
    cuda = device.create_cuda_gpu_on(1)

    net = SANet('sanet', cuda, num_stack_layers=args.s, embed_size=args.e, hidden_size=args.d,
        seq_length=args.l, vocab_size=args.v, batchsize=args.b)
    net.create_net(dropout=0.2)
    net.train(args.data, args.m)
    #net.inference(args.data, batchsize=1)
