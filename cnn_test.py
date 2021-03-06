import cPickle as pickle
import numpy as np
import argparse

from singa import layer
from singa.layer import Layer
from singa import loss
from singa import device
from singa import tensor
from singa import optimizer
from singa import initializer
from singa.proto import model_pb2, core_pb2
from singa import utils
from singa import net as ffnet
from singa import metric
from time import time

from data_loader import *


def convert_sample(data, seq_length, vocab_size, dev):
    '''convert a batch of data into a sequence of input tensors'''
    x = np.zeros((seq_length, vocab_size), dtype=np.float32)
    for t in range(seq_length):
        x[t, data[t]] = 1
    return x

def convert_samples(data, seq_length, vocab_size, dev):
    x = []
    for t in list(data):
        x.append(convert_sample(t, seq_length, vocab_size, dev))
    return np.array(x)

def multi_tuple(t):
    multi = 1
    for i in range(len(t)):
        multi *= t[i]
    return multi

def get_lr(epoch):
    return 0.01 / float(1 << (epoch / 50))

class Reshape(Layer):
    '''
    Reshape the input tensor
    '''
    def __init__(self, name, output_shape, input_sample_shape=None):
        '''
        Args:
            output_shape: a tuple of output shape
            input_sample_shape: a tuple of input shape
        '''
        self.output_shape = output_shape
        self.output_size = multi_tuple(self.output_shape)
        super(Reshape, self).__init__(name)
        if input_sample_shape is not None:
            self.setup(input_sample_shape)

    def setup(self, in_shapes):
        self.in_shape = in_shapes
        self.has_setup=True

    def get_output_sample_shape(self):
        return self.output_shape

    def forward(self, flag, input):
        ''' Reshape the input tensor into output_shape

        Args:
             flag: not used
             input: a single input tensor

        Returns:
            output tensor (reshaped input)
        '''
        assert isinstance(input, tensor.Tensor), 'The input must be Tensor'
        self.in_shape = input
        outputs = tensor.reshape(input, tuple(list([(self.in_shape.size() / self.output_size)])) + self.output_shape)
        return outputs

    def backward(self, flag, dy):
        ''' Return gradient, []'''
        dx = tensor.reshape(dy, self.in_shape.shape)
        return dx, []


class CNNNet(object):
    def __init__(self, embed_size=128, maxlen=53, max_features=33366, kernel_size=5, filters=64, pool_size=4, batch_size=32,  use_cpu=True):
        self.embed_size = embed_size  # word_embedding size
        self.maxlen = maxlen  # used to pad input tweet sequence
        self.vocab_size = max_features  # vocabulary size
        self.batch_size = batch_size
        # cnn
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size

        self.use_cpu = use_cpu

    def build_net(self):
        if self.use_cpu:
            layer.engine = 'singacpp'
        else:
            layer.engine = 'cudnn'
        self.net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
        self.net.add(Reshape('reshape1', (self.vocab_size,), input_sample_shape=(self.maxlen, self.vocab_size)))
        self.net.add(layer.Dense('embed', self.embed_size))  # output: (embed_size, )
        self.net.add(layer.Dropout('dropout'))
        self.net.add(Reshape('reshape2', (1, self.maxlen, self.embed_size)))
        self.net.add(layer.Conv2D('conv', self.filters, (self.kernel_size, self.embed_size), border_mode='valid'))  # output: (filter, embed_size)
        if self.use_cpu==False:
            self.net.add(layer.BatchNormalization('batchNorm'))
        self.net.add(layer.Activation('activ'))  # output: (filter, embed_size)
        self.net.add(layer.MaxPooling2D('max', stride = self.pool_size))
        self.net.add(layer.Flatten('flatten'))
        self.net.add(layer.Dense('dense', 2))

    def train(self, data, max_epoch, model_path = 'model'):
        if self.use_cpu:
            print 'Using CPU'
            self.dev = device.get_default_device()
        else:
            print 'Using GPU'
            self.dev = device.create_cuda_gpu()

        self.net.to_device(self.dev)
        opt = optimizer.SGD(momentum=0.9, weight_decay=1e-4)
        # opt = optimizer.RMSProp(constraint=optimizer.L2Constraint(5))
        for (p, n) in zip(self.net.param_values(), self.net.param_names()):
            if 'var' in n:
                p.set_value(1.0)
            elif 'gamma' in n:
                p.uniform(0, 1)
            elif 'weight' in n:
                p.gaussian(0, 0.01)
            else:
                p.set_value(0.0)
            print n, p.shape, p.l1()

        tx = tensor.Tensor((self.batch_size, self.maxlen, self.vocab_size), self.dev)
        ty = tensor.Tensor((self.batch_size,), self.dev, core_pb2.kInt)
        train_x, train_y, test_x, test_y = data
        num_train_batch = train_x.shape[0] / self.batch_size
        num_test_batch = test_x.shape[0] / self.batch_size
        idx = np.arange(train_x.shape[0], dtype=np.int32)
        for epoch in range(max_epoch):
            np.random.shuffle(idx)
            loss, acc = 0.0, 0.0
            print '\nEpoch %d' % epoch
            start = time()
            for b in range(num_train_batch):
                batch_loss, batch_acc = 0.0, 0.0
                grads = []
                x = train_x[idx[b * self.batch_size: (b + 1) * self.batch_size]]  # x.shape = (batch_size, maxlen)
                y = train_y[idx[b * self.batch_size: (b + 1) * self.batch_size]]  # y.shape = (batch_size,)
                # for input as (batch_size, max_len, vocab_size)
                sam_arrs = convert_samples(x, x.shape[1], self.vocab_size, self.dev)
                tx.copy_from_numpy(sam_arrs)
                ty.copy_from_numpy(np.array(y, dtype='int32'))
                grads, (batch_loss, batch_acc) = self.net.train(tx,ty)
                for (s, p, g) in zip(self.net.param_names(), self.net.param_values(), grads):
                    opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
                # update progress bar
                utils.update_progress(b * 1.0 / num_train_batch,
                                      'training loss = %f, accuracy = %f' % (batch_loss, batch_acc))
                loss += batch_loss
                acc += batch_acc

            print "\ntraining time = ", time() - start
            info = 'training loss = %f, training accuracy = %f, lr = %f' \
                   % (loss / num_train_batch, acc / num_train_batch, get_lr(epoch))
            print info

            loss, acc = 0.0, 0.0
            start = time()
            for b in range(num_test_batch):
                batch_loss, batch_acc = 0.0, 0.0
                x = test_x[b * self.batch_size: (b + 1) * self.batch_size]  # x.shape = (batch_size, maxlen)
                y = test_y[b * self.batch_size: (b + 1) * self.batch_size]
                sam_arrs = convert_samples(x, x.shape[1], self.vocab_size, self.dev)
                tx.copy_from_numpy(sam_arrs)
                ty.copy_from_numpy(np.array(y, dtype='int32'))
                grads, (batch_loss, batch_acc) = self.net.train(tx, ty)
                loss += batch_loss
                acc += batch_acc

            print "evaluation time = ", time() - start
            print 'test loss = %f, test accuracy = %f \n' \
                  % (loss / num_test_batch, acc / num_test_batch)

            if (epoch % 2) == 1 or epoch + 1 == max_epoch:
                # checkpoint the file model
                with open('%s_%d.bin' % (model_path, epoch), 'wb') as fd:
                    print 'saving model to %s_%d.bin' % (model_path, epoch)
                    d = {}
                    for name, w in zip( self.net.param_names(), self.net.param_values()):
                        w.to_host()
                        d[name] = tensor.to_numpy(w)
                        w.to_device(self.dev)
                    pickle.dump(d, fd)

    def load(self, model_path):
        with open(model_path, 'rb') as fd:
            params = pickle.load(fd)

        if self.net:
            for name, w in zip( self.net.param_names(), self.net.param_values() ):
                w.copy_from_numpy(params[name])
        else:
            print 'please build net first.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-stack LSTM for '
        'modeling  character sequence from plain text files')
    parser.add_argument('-b', type=int, default=32, help='batch_size')
    parser.add_argument('-l', type=int, default=53, help='sequence length')
    parser.add_argument('-e', type=int, default=128, help='embed size')
    parser.add_argument('-k', type=int, default=5, help='kernel size')
    parser.add_argument('-f', type=int, default=64, help='num of filters')
    parser.add_argument('-p', type=int, default=4, help='pooling size')
    parser.add_argument('-m', type=int, default=10, help='max num of epoch')
    parser.add_argument('-v', type=int, default=33366, help='vocabulary size')
    parser.add_argument('-c', type=int, default=True, help='CPU flag')
    args = parser.parse_args()
    #train_dat, train_label, val_dat, val_label = load_sample()
    train_dat, train_label, val_dat, val_label = load_corpus('dataset.pkl')
    print 'train_dat.shape = ', train_dat.shape ,'train_label.shape = ', train_label.shape
    print 'val_dat.shape = ', val_dat.shape ,'val_label.shape = ', val_label.shape
    data = (train_dat, train_label, val_dat, val_label)
    n = CNNNet(embed_size=args.e, maxlen=args.l, max_features=args.v, kernel_size=args.k, filters=args.f, pool_size=args.p, batch_size=args.b,  use_cpu=args.c)
    n.build_net()
    n.train(data, args.m)

