import numpy as np
from lib.functions import reorderConsensusMatrix
import pandas as pd


# Classifies along the columns
def classify_by_max(x: np.array):
    return (x == np.amax(x, axis=0)).astype(float)


def classify_by_z(x: np.array, thresh):
    a = (x - np.mean(x, axis=1).reshape((-1, 1))) / (np.std(x, axis=1)).reshape((-1, 1))
    classification = np.zeros(a.shape)
    classification[a > thresh] = 1
    return classification


class NmfModel:

    def __init__(self, a: pd.DataFrame, k: int, niter: int, super_niter: int):
        self.k = k
        self.x_df = a
        self.x = a.values

        self.niter = niter
        self.super_niter = super_niter
        self.eps = np.finfo(self.x.dtype).eps

        self.error = float('inf')

    def initialize_variables(self):
        self.cmh = np.zeros((self.x.shape[1], self.x.shape[1]))
        self.cmw = np.zeros((self.x.shape[0], self.x.shape[0]))

    def initialize_wh(self):
        self.w = np.random.rand(self.x.shape[0], self.k)
        self.h = np.random.rand(self.k, self.x.shape[1])

    # TODO - Try masking
    def update_weights(self):
        w = self.w
        h = self.h
        self.w = w * (np.dot(self.x / (self.eps + np.dot(w, h)), h.T) / np.sum(h, 1))
        self.h = h * (np.dot(w.T, self.x / (self.eps + np.dot(w, h))) / np.sum(w, 0).reshape((-1, 1)))
        self.calc_error()

    def wrapper_update(self, verbose=0):
        for i in range(1, self.niter):
            self.update_weights()
            self.calc_error()
            if verbose == 1 and i % 10 == 0:
                print("\t\titer: %i | error: %f" % (i, self.error))

    def super_wrapper(self, verbose=0):
        self.initialize_variables()
        for i in range(0, self.super_niter):
            self.initialize_wh()

            if verbose == 1 and i % self.super_niter % 1 == 0:
                print("\tSuper iteration: %i Error: %f " % (i, self.error))
                self.wrapper_update(verbose=1)
            else:
                self.wrapper_update(verbose=0)

            self.cmh += np.dot(classify_by_max(self.h).T, classify_by_max(self.h))
            self.cmw += np.dot(classify_by_max(self.w.T).T, classify_by_max(self.w.T))

        class_list = ["class-%i" % a for a in list(range(self.k))]

        self.h = pd.DataFrame(self.h, columns=self.x_df.columns, index=class_list)
        self.cmh /= self.super_niter
        self.cmh = reorderConsensusMatrix(self.cmh)

        self.w = pd.DataFrame(self.w, index=self.x_df.index, columns=class_list)
        self.cmw /= self.super_niter
        self.cmw = reorderConsensusMatrix(self.cmw)

    def calc_error(self):
        self.error = np.mean(np.abs(self.x - np.dot(self.w, self.h)))

    def predicted_matrix(self):
        return np.dot(self.w, self.h)

