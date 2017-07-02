import numpy as np
import time


class Embedding():
    def __init__(self, name, dim, voc_size, std=0.01):
        self.dim = dim
        self.x = None
        self.w = np.array(np.random.normal(0, std, (voc_size, dim)), dtype=np.float32)
        self.gw = np.zeros(self.w.shape, dtype=np.float32)
        self.w_hist = np.zeros(self.w.shape, dtype=np.float32)

    def forward(self, flag, x):
        if flag:
            self.x = x
        out = np.empty((x.shape[0], 1, x.shape[1], self.dim), dtype=np.float32)
        for sid in range(x.shape[0]):
            for tid in range(x.shape[1]):
                wid = x[sid, tid]
                out[sid, 0, tid] = self.w[wid]
        return out

    def backward(self, dy):
        wset = set([])
        dy = dy.reshape((dy.shape[0], dy.shape[2], dy.shape[3]))
        for sid in range(dy.shape[0]):
            for tid in range(dy.shape[1]):
                wid = self.x[sid, tid]
                self.gw[wid] += dy[sid, tid]
                wset.add(wid)
        self.wlist = list(wset)

    def update(self, lr, mom, decay, nrm2=0):
        idx = np.array(self.wlist)
        self.w_hist[idx] *= mom
        # self.gw[idx] *= decay
        self.w_hist[idx] -= lr * self.gw[idx]
        self.w[idx] += self.w_hist[idx]
        self.gw[idx] *= 0
        if nrm2 > 0:
            nrm = np.linalg.norm(self.w[idx], axis=1)
            r = nrm > nrm2
            if np.sum(r) > 0:
                idx = idx[r]
                scale = nrm2 / nrm[idx]
                self.w[idx] *= scale[np.newaxis, :]


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
