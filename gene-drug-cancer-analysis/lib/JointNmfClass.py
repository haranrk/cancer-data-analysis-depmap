from lib.functions import *
import random


def classify_by_max(x: np.array):
    return (x == np.amax(x, axis=0)).astype(float)


def classify_by_z(x: np.array, thresh):
    a = (x - np.mean(x, axis=1).reshape((-1, 1))) / (np.std(x, axis=1)).reshape((-1, 1))
    classification = np.zeros(a.shape)
    classification[a > thresh] = 1
    return classification


# Abstract Class - Do not instantiate this class
# Returns all the matrices as a DataFrame
class JointNmfClass:
    def __init__(self, x: dict, k: int, niter: int, super_niter: int, thresh: int):
        if str(type(list(x.values())[0])) == "<class 'pandas.core.frame.DataFrame'>":
            self.x = {k: x[k].values for k in x}
            self.x_df = x
        elif str(type(list(x.values())[0])) == "<class 'numpy.ndarray'>":
            self.x = x
            self.x_df = pd.DataFrame(x)
        else:
            raise ValueError("Invalid DataType")

        self.k = k
        self.niter = niter
        self.super_niter = super_niter
        self.cmw = None
        self.w = None
        self.h = None
        self.z_score = None
        self.thresh = thresh
        self.error = float('inf')
        self.eps = np.finfo(list(self.x.values())[0].dtype).eps

    def initialize_variables(self):
        number_of_samples = list(self.x.values())[0].shape[0]

        self.cmw = np.zeros((number_of_samples, number_of_samples))
        # self.w_avg = np.zeros((number_of_samples, self.k))

        # self.h_avg = {}
        self.max_class = {}
        self.max_class_cm = {}
        self.z_class = {}
        self.z_class_cm = {}
        self.z_score = {}

        for key in self.x:
            number_of_features = self.x[key].shape[1]
            # self.h_avg[key] = np.zeros((self.k, number_of_features))
            self.max_class[key] = np.zeros((self.k, number_of_features))
            self.max_class_cm[key] = np.zeros((number_of_features, number_of_features))
            self.z_class[key] = np.zeros((self.k, number_of_features))
            self.z_class_cm[key] = np.zeros((number_of_features, number_of_features))
            self.z_score[key] = np.zeros((self.k, number_of_features))

    def wrapper_update(self, verbose=0):
        for i in range(1, self.niter):
            self.update_weights()
            # for key in self.x:
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

            self.cmw += self.connectivity_matrix_w()
            # self.w_avg += self.w

            for key in self.h:
                # self.h_avg[key] += self.h[key]
                connectivity_matrix = lambda a: np.dot(a.T, a)
                self.max_class_cm[key] += connectivity_matrix(classify_by_max(self.h[key]))
                self.z_class_cm[key] += connectivity_matrix(classify_by_z(self.h[key], self.thresh))

        # Normalization
        # self.w_avg = self.w_avg/self.super_niter
        self.cmw = reorderConsensusMatrix(self.cmw / self.super_niter)

        for key in self.h:
            # self.h_avg[key] /= self.super_niter
            self.max_class_cm[key] /= self.super_niter
            self.z_class_cm[key] /= self.super_niter
            self.max_class_cm[key] = reorderConsensusMatrix(self.max_class_cm[key])
            # self.z_class_cm[key] = reorderConsensusMatrix(self.z_class_cm[key])
        self.calc_z_score()

        # Classification
        for key, val in self.h.items():
            self.max_class[key] = classify_by_max(val)
            self.z_class[key] = classify_by_z(val, self.thresh)

        # Converting values to DataFrames
        class_list = ["class-%i" % a for a in list(range(self.k))]
        # noinspection PyCallingNonCallable
        self.w = pd.DataFrame(self.w, index=random.choice(list(self.x_df.values())).index, columns=class_list)

        self.h = self.conv_dict_np_to_df(self.h)
        # self.h_avg = self.conv_dict_np_to_df(self.h_avg)
        self.z_score = self.conv_dict_np_to_df(self.z_score)
        self.max_class = self.conv_dict_np_to_df(self.max_class)
        self.z_class = self.conv_dict_np_to_df(self.z_class)

    def conv_dict_np_to_df(self, a: dict):
        class_list = ["class-%i" % a for a in list(range(self.k))]
        return {k: pd.DataFrame(a[k], index=class_list, columns=self.x_df[k].columns) for k in a}

    # TODO - invalid value ocurred
    def calc_z_score(self):
        for key in self.h:
            self.z_score[key] = (self.h[key] - np.mean(self.h[key], axis=1).reshape((-1, 1))) / (
                    self.eps + np.std(self.h[key], axis=1).reshape((-1, 1)))

    def connectivity_matrix_w(self):
        max_tiled = np.tile(self.w.max(1).reshape((-1, 1)), (1, self.w.shape[1]))
        max_index = np.zeros(self.w.shape)
        max_index[self.w == max_tiled] = 1
        return np.dot(max_index, max_index.T)

    def calc_error(self):
        self.error = 0
        for key in self.x:
            self.error += np.mean(np.abs(self.x[key] - np.dot(self.w, self.h[key])))

    def update_weights(self):
        raise NotImplementedError("Must override update_weights")

    def initialize_wh(self):
        raise NotImplementedError("Must override initialize_wh")
