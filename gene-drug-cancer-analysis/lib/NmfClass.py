import numpy as np
import pandas as pd
from lib.functions import *
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform





class NmfModel:

    def __init__(self, a: np.array, k: int, niter: int, super_niter: int):
        self.k = k
        self.X = a
        self.w = np.zeros((self.X.shape[0], self.k))
        self.h = np.zeros((self.k, self.X.shape[1]))

        self.niter = niter
        self.super_niter = super_niter
        self.eps = np.finfo(a.dtype).eps

        self.initialize_w_h()
        self.error = np.mean(np.abs(a - np.dot(self.w, self.h)))
        self.consensus_matrix_h = np.zeros((self.X.shape[1], self.X.shape[1]))
        self.consensus_matrix_w = np.zeros((self.X.shape[0], self.X.shape[0]))

    def initialize_w_h(self):
        self.w = np.random.rand(self.X.shape[0], self.k)
        self.h = np.random.rand(self.k, self.X.shape[1])

    # TODO - Try masking
    def update_weights(self):
        w = self.w
        h = self.h
        self.w = w * (np.dot(self.X / (self.eps + np.dot(w, h)), h.T) / np.sum(h, 1))
        self.h = h * (np.dot(w.T, self.X / (self.eps + np.dot(w, h))) / np.sum(w, 0).reshape((-1, 1)))
        self.calc_error()

    def wrapper_update(self, verbose=0):
        for x in range(1, self.niter):
            self.update_weights()
            if verbose == 1:
                print(self.error)

    def super_wrapper(self, verbose=0):
        for i in range(0, self.super_niter):
            self.initialize_w_h()
            self.wrapper_update(verbose=0)
            if verbose == 1 and i % 5 == 0:
                print("Super iteration: %i Error: %f " % (i, self.error))
            self.consensus_matrix_h += self.connectivity_matrix_h()
            self.consensus_matrix_w += self.connectivity_matrix_w()
        self.consensus_matrix_h /= self.super_niter
        self.consensus_matrix_w /= self.super_niter
        self.consensus_matrix_w = reorderConsensusMatrix(self.consensus_matrix_w)
        self.consensus_matrix_h = reorderConsensusMatrix(self.consensus_matrix_h)

    def calc_error(self):
        self.error = np.mean(np.abs(self.X - np.dot(self.w, self.h)))

    def predicted_matrix(self):
        return np.dot(self.w, self.h)

    def cluster(self):
        return np.argmax(self.h, 0)

    def connectivity_matrix_h(self):
        max_tiled = np.tile(self.h.max(0), (self.h.shape[0], 1))
        max_index = np.zeros(self.h.shape)
        max_index[self.h == max_tiled] = 1
        return np.dot(max_index.T, max_index)

    def connectivity_matrix_w(self):
        max_tiled = np.tile(self.w.max(1).reshape((-1, 1)), (1, self.w.shape[1]))
        max_index = np.zeros(self.w.shape)
        max_index[self.w == max_tiled] = 1
        return np.dot(max_index, max_index.T)

