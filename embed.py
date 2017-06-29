import numpy as np
import time

class Embedding():
    def __init__(self, name, dim, voc_size, std=0.01):
        self.dim = dim
        self.x = None
        self.w = np.random.normal(0, std, (voc_size, dim))
        self.gw = np.zeros(self.w.shape)
        self.w_hist = np.zeros(self.w.shape)

    def forward(self, flag, x):
        if flag:
            self.x = x
        out = np.empty((x.shape[0], x.shape[1], self.dim))
        for sid in range(x.shape[0]):
            for tid in range(x.shape[1]):
                wid = x[sid, tid]
                out[sid, tid] = self.w[wid]
        return out

    def backward(self, dy):
        wset = set([])
        for sid in range(x.shape[0]):
            for tid in range(x.shape[1]):
                wid = x[sid, tid]
                self.gw[wid] += dy[sid, tid]
                wset.add(wid)
        self.wlist = list(wset)

    def update(self, lr, mom, decay, nrm2=0):
        idx = self.wlist
        self.w_hist[idx] *= mom
        self.gw[idx] *= decay
        self.w_hist[idx] -= lr * self.gw[idx]
        self.w[idx] += self.w_hist[idx]
        self.gw[idx] *= 0
        if nrm2 > 0:
            nrm = np.linalg.norm(self.w[idx])
            r = nrm > nrm2
            idx = idx[r]
            self.w[idx] *= nrm2 / nrm[idx]


if __name__ == '__main__':
    e = Embedding('embed', 128, 33366)
    start = time.time()
    x = np.random.randint(0, 33366, (32, 53))
    dy = np.random.normal(0, 0.001, (32, 53, 128))
    for i in range(100):
        y = e.forward(True, x)
        e.backward(dy)
        e.update(0.01, 0.9, 1e-4)
    print('time per iter = %f' % ((time.time() - start)/float(100 * 32)))
