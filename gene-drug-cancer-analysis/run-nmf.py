import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import argparse as ap

import pandas as pd
import numpy as np
from lib.functions import clean_df
from lib.IntegrativeJnmfClass import IntegrativeNmfClass
from lib.NmfClass import NmfModel
import os
from pathlib import Path as pth
main_dir = pth(os.getcwd()).resolve()
script_dir = pth(__file__).parent.absolute()
os.chdir(script_dir)
print(os.getcwd())
print(main_dir)
print(script_dir)

parser = ap.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
parser.add_argument("data_name", type=str, help="Which dataset to use", choices=["Avana", "GeCKO", "RNAi_Ach", "RNAi_merged", "RNAi_Nov_DEM"])
parser.add_argument("k", type=int, help="rank according to which the matrix will be factorized")
parser.add_argument("iter",type=int, help="the maximum number of iterations that the nmf will run for" )
parser.add_argument("trials", type=int, help="number of trails against which the consensus data will be plotted")
args = parser.parse_args()

print(args.data_name)
a = pd.read_csv("data/%s.csv" % args.data_name, index_col=0)
print(a.shape)
a = clean_df(a, axis=1)
a = (a - (np.min(a.values))) / np.std(a.values)
data = {args.data_name:a}
print(data[args.data_name].shape)

k = args.k
niter = args.iter
super_niter = args.trials

print("Rank: %i | iterations: %i | trials: %i" % (k, niter, super_niter))
m = IntegrativeNmfClass(data, k, niter, super_niter, lamb=5, thresh=0.1)
m.super_wrapper(verbose=args.verbose)
plt.figure()
plt.subplot(211)
plt.title("cmh")
sns.heatmap(m.max_class_cm[args.data_name])
plt.title("cmw")
plt.subplot(212)
sns.heatmap(m.cmw)
os.chdir(main_dir)
plt.savefig("%s_%i_%i_%i" % (args.data_name, k, niter, super_niter))
