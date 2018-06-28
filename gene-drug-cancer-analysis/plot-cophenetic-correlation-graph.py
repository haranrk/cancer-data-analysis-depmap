import argparse as ap
import pandas as pd
from lib.IntegrativeJnmfClass import IntegrativeNmfClass
from matplotlib import pyplot as plt
import os
import seaborn as sns
from lib.functions import clean_df
import numpy as np
from pathlib import Path as pth
main_dir = pth(os.getcwd()).resolve()
script_dir = pth(__file__).parent.absolute()
os.chdir(script_dir)

parser = ap.ArgumentParser()
parser.add_argument("data_name", type=str, help="Which dataset to use", choices=["Avana", "GeCKO", "RNAi_Ach", "RNAi_merged", "RNAi_Nov_DEM","filtered_avana", "filtered_nov_dem"])
parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
parser.add_argument("k", type=int, help="rank uptil which the copehenetic correlation code must be plotted")
parser.add_argument("iter",type=int, help="the maximum number of iterations that the nmf will run for" )
parser.add_argument("trials", type=int, help="number of trails against which the consensus data will be plotted")
args = parser.parse_args()

print(args.data_name)
a = pd.read_csv("data/%s.csv" % args.data_name, index_col=0)
print("Original shape: %s"%str(a.shape))
a = clean_df(a, axis=1)
a = (a - (np.min(a.values))) / np.std(a.values)
data = {args.data_name:a}
print("Cleaned shape: %s"%str(data[args.data_name].shape))

k_list = list(range(2,args.k))
niter = args.iter
super_niter = args.trials
coph_corr_list_w = []
coph_corr_list_h = []
for k in k_list:
    print("Rank: %i | iterations: %i | trials: %i" % (k, niter, super_niter))
    m = IntegrativeNmfClass(data, k, niter, super_niter, lamb=5, thresh=0.1)
    m.super_wrapper(verbose=args.verbose)
    coph_corr_list_w.append(m.coph_corr_w)
    coph_corr_list_h.append(m.coph_corr_h[args.data_name])

os.chdir(main_dir)
plt.figure(figsize=(24, 18))
plt.suptitle("Cophenetic Correlation Plot for: %s"%args.data_name)
plt.subplot(212)
plt.title("w")
plt.scatter(k_list,coph_corr_list_w)
for i in range(0,len(k_list)):
    plt.annotate("k=%i | corr=%0.2f"%(k_list[i], coph_corr_list_w[i]),xy=((k_list[i], coph_corr_list_w[i])), xytext=((k_list[i], coph_corr_list_w[i]+0.01)) )
plt.plot(k_list,coph_corr_list_w)

plt.subplot(211)
plt.title("h")
plt.plot(k_list,coph_corr_list_h)
plt.scatter(k_list,coph_corr_list_h)
for i in range(0,len(k_list)):
    plt.annotate("k=%i | corr=%0.2f"%(k_list[i], coph_corr_list_h[i]),((k_list[i], coph_corr_list_h[i])), xytext=((k_list[i], coph_corr_list_h[i]+0.01)) )
plt.savefig("coph_corr_%s" % (args.data_name))
    