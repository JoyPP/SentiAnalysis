import cPickle as pickle
import numpy as np

from singa import layer
from singa import loss
from singa import device
from singa import tensor
from singa import optimizer
from singa import initializer
from singa.proto import model_pb2, core_pb2
from singa import utils
from singa import net as ffnet
from singa import metric

from data_loader import *


def convert_sample(data, seq_length, vocab_size, dev):
    '''convert a batch of data into a sequence of input tensors'''
    x = np.zeros((seq_length, vocab_size), dtype=np.float32)
    for t in range(seq_length):
        x[t, data[t]] = 1
    return x


def get_lr(epoch):
    return 0.01 / float(1 << (epoch / 50))


class CNNNet(object):
    def __init__(self, embed_size=128, maxlen=53, max_features=33366, kernel_size=5, filters=64, pool_size=4, batch_size=32):
        self.embed_size = embed_size  # word_embedding size
        self.maxlen = maxlen  # used to pad input tweet sequence
        self.vocab_size = max_features  # vocabulary size
        self.batch_size = batch_size
        # cnn
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size

    def build_net(self):
        layer.engine = 'singacpp'
        self.net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
        self.net.add(
            layer.Dense('embed', self.embed_size, input_sample_shape=(self.vocab_size,)))  # output: (embed_size, )
        self.net.add(layer.Conv2D('conv', self.filters, (self.kernel_size, self.embed_size), border_mode='valid',
                                  input_sample_shape=(self.batch_size, 1, self.maxlen, self.embed_size,)))  # output: (filter, embed_size)
        self.net.add(layer.Activation('activ'))  # output: (filter, embed_size)
        self.net.add(layer.MaxPooling2D('max', self.kernel_size, self.pool_size))
        self.net.add(layer.Flatten('flatten'))
        self.net.add(layer.Dense('dense', 2))

    def train(self, data, max_epoch, use_cpu=True):
        if use_cpu:
            print 'Using CPU'
            dev = device.get_default_device()
        else:
            print 'Using GPU'
            dev = device.create_cuda_gpu()

        self.net.to_device(dev)
        opt = optimizer.SGD(momentum=0.9, weight_decay=1e-4)
        # opt = optimizer.RMSProp(constraint=optimizer.L2Constraint(5))
        for (p, n) in zip(self.net.param_values(), self.net.param_names()):
            if 'weight' in n:
                p.gaussian(0, 0.01)
            else:
                p.set_value(0)

        tx = tensor.Tensor((self.maxlen, self.vocab_size), dev)
        ty = tensor.Tensor((1,), dev, core_pb2.kInt)
        train_x, train_y, test_x, test_y = data
        num_train_batch = train_x.shape[0] / self.batch_size
        num_test_batch = test_x.shape[0] / self.batch_size
        idx = np.arange(train_x.shape[0], dtype=np.int32)
        for epoch in range(max_epoch):
            np.random.shuffle(idx)
            loss, acc = 0.0, 0.0
            print 'Epoch %d' % epoch
            for b in range(num_train_batch):
                batch_loss, batch_acc = 0.0, 0.0
                grads = []
                x = train_x[idx[b * self.batch_size: (b + 1) * self.batch_size]]  # x.shape = (batch_size, maxlen)
                y = train_y[idx[b * self.batch_size: (b + 1) * self.batch_size]]  # y.shape = (batch_size,)
                for idx_sam in range(len(x)):
                    sam_arr = x[idx_sam]  # one sequence (maxlen,)
                    sam_arr = convert_sample(sam_arr, sam_arr.shape[0], self.vocab_size, dev)
                    tx = tensor.from_numpy(sam_arr)
                    ty = tensor.from_numpy(y[idx_sam:idx_sam+1])
                    grad, (l, a) = self.net.train(tx, ty)
                    batch_loss += l
                    batch_acc += a
                    grads += grad
                for (s, p, g) in zip(self.net.param_names(), self.net.param_values(), grads):
                    opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
                # update progress bar
                utils.update_progress(b * 1.0 / num_train_batch,
                                      'training loss = %f, accuracy = %f' % (batch_loss/num_train_batch, batch_acc/num_train_batch))
                loss += batch_loss
                acc += batch_acc

            info = '\n training loss = %f, training accuracy = %f, lr = %f' \
                   % (loss / num_train_batch, acc / num_train_batch)
            print info

            loss, acc = 0.0, 0.0
            for b in range(num_test_batch):
                batch_loss, batch_acc = 0.0, 0.0
                x = test_x[idx[b * batch_size: (b + 1) * batch_size]]  # x.shape = (batch_size, maxlen)
                y = test_y[idx[b * batch_size: (b + 1) * batch_size]]
                for idx_sam in range(len(x)):
                    sam_arr = x[idx_sam]
                    sam_arr = convert_sample(sam_arr, sam_arr.shape[0], self.vocab_size, dev)
                    tx.copy_from_numpy(sam_arr)
                    ty.copy_from_numpy(y[idx_sam])
                    grad, (l, a) = self.net.train(tx, ty)
                    batch_loss += l
                    batch_acc += a
                loss += batch_loss
                acc += batch_acc

            print 'test loss = %f, test accuracy = %f' \
                  % (loss / num_train_batch, acc / num_train_batch)


if __name__ == '__main__':
    train_dat, train_label, val_dat, val_label = load_sample()
    data = (train_dat, train_label, val_dat, val_label)
    n = CNNNet()
    n.build_net()
    n.train(data, 5)