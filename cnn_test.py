import numpy as np
import argparse
from tqdm import trange

from singa.layer import Conv2D, MaxPooling2D, Activation, BatchNormalization, \
    Dense, Flatten, Dropout
from singa.loss import SoftmaxCrossEntropy
from singa.metric import Accuracy
from singa import device
from singa import tensor
from singa import optimizer
from singa import net as ffnet

import math
from data_loader import *
from embed import Embedding


#ffnet.verbose=True


def get_lr(epoch, cfg):
    return cfg.lr * math.pow(0.1, float(epoch // cfg.decay_freq))


def update_perf(his, cur, a=0.8):
    '''Accumulate the performance by considering history and current values.'''
    return his * a + cur * (1 - a)


class CNNNet(object):
    def __init__(self, cfg):
        self.cfg = cfg
        net = ffnet.FeedForwardNet()
        self.embed = Embedding('embed', cfg.embed_size, cfg.vocab_size)
        net.add(Conv2D('conv1', 32, (3, cfg.embed_size), pad=0,
                       input_sample_shape=(1, cfg.seq_length, cfg.embed_size)))
        net.add(BatchNormalization('bn1'))
        net.add(Activation('activ1'))
        #net.add(Conv2D('conv2', 32, (3, 1), pad=0))
        #net.add(BatchNormalization('bn2'))
        #net.add(Activation('activ2'))
        net.add(MaxPooling2D('pool1', (3, 1), (2, 1), pad=0))
        net.add(Conv2D('conv3', 32, (3, 1), pad=0))
        net.add(BatchNormalization('bn3'))
        net.add(Activation('activ3'))
        net.add(MaxPooling2D('pool2', (3, 1), (2, 1), pad=0))
        net.add(Flatten('flatten'))
        net.add(Dense('dense1', 10))
        net.add(Activation('actv4'))
        net.add(Dropout('drop'))
        net.add(Dense('dense2', 2))
        self.net = net


    def train(self, data, cfg, dev):
        opt = optimizer.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)
        loss = SoftmaxCrossEntropy()
        acc = Accuracy()
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
        self.net.to_device(dev)

        tx = tensor.Tensor((cfg.batch_size, 1, cfg.seq_length, cfg.embed_size), dev)
        ty = tensor.Tensor((cfg.batch_size,), dev, tensor.int32)
        train_x, train_y, test_x, test_y = data
        print train_x.shape, train_y.shape
        num_train_batch = train_x.shape[0] / cfg.batch_size
        num_test_batch = test_x.shape[0] / cfg.batch_size
        idx = np.arange(train_x.shape[0], dtype=np.int32)
        for epoch in range(cfg.max_epoch):
            bar = trange(num_train_batch, desc='Epoch %d, Train' % epoch)
            meter = {'loss': 0, 'acc':0}
            np.random.shuffle(idx)
            for b in bar:
                x = train_x[idx[b * cfg.batch_size: (b + 1) * cfg.batch_size]]
                y = train_y[idx[b * cfg.batch_size: (b + 1) * cfg.batch_size]]
                # for input as (batch_size, max_len, vocab_size)
                ty.copy_from_numpy(np.array(y, dtype='int32'))
                e = self.embed.forward(True, x)
                tx.copy_from_numpy(e)
                logits = self.net.forward(True, tx)
                meter['loss'] = update_perf(meter['loss'], loss.forward(True, logits, ty).l1())
                meter['acc'] = update_perf(meter['acc'], acc.forward(logits, ty).l1())
                grad = loss.backward()
                embed_grad=None
                for (slist, plist, glist, ret) in self.net.backward(grad, output='conv1'):
                    for s, p, g in zip(slist, plist, glist):
                        opt.apply_with_lr(epoch, get_lr(epoch, cfg), g, p, str(s))
                    if 'conv1' in ret :
                        embed_grad = ret['conv1']
                self.embed.backward(tensor.to_numpy(embed_grad))
                self.embed.update(get_lr(epoch, cfg), cfg.mom, cfg.weight_decay, cfg.nrm2)
                # update progress bar
                bar.set_postfix(**meter)
            bar.close()


            meter = {'loss':0.0, 'acc': 0.0}
            for b in range(num_test_batch):
                x = test_x[b * cfg.batch_size: (b + 1) * cfg.batch_size]
                y = test_y[b * cfg.batch_size: (b + 1) * cfg.batch_size]
                ty.copy_from_numpy(np.array(y, dtype='int32'))
                tx.copy_from_numpy(self.embed.forward(False, x))
                logits = self.net.forward(False, tx)
                meter['loss'] += loss.forward(False, logits, ty).l1()
                meter['acc'] += acc.forward(logits, ty).l1()
            print('Evaluation at Epoch = %d, loss=%f, acc=%f' %
                  (epoch, meter['loss']/num_test_batch,
                   meter['acc']/num_test_batch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-stack LSTM for '
        'modeling  character sequence from plain text files')
    parser.add_argument('--corpus', default='/home/piaopiao/bnp/SentiAnalysis/dataset.pkl')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--seq_length', type=int, default=53, help='sequence length')
    parser.add_argument('--embed_size', type=int, default=32, help='embed size')
    parser.add_argument('--max_epoch', type=int, default=10, help='max num of epoch')
    parser.add_argument('--vocab_size', type=int, default=33366, help='vocabulary size')
    parser.add_argument('--decay_freq', type=int, default=10, help='learning rate decay')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--nrm2', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    # train_dat, train_label, val_dat, val_label = load_sample(args.corpus)
    train_dat, train_label, val_dat, val_label = load_corpus(args.corpus, sample_prec=1)
    print 'train_dat.shape = ', train_dat.shape ,'train_label.shape = ', train_label.shape
    print 'val_dat.shape = ', val_dat.shape ,'val_label.shape = ', val_label.shape
    data = (train_dat, train_label, val_dat, val_label)
    net = CNNNet(args)
    dev = device.create_cuda_gpu()
    net.train(data, args, dev)
