import subprocess as sp
import os 
from pathlib import Path as pth
main_dir = pth(os.getcwd()).resolve()
script_dir = pth(__file__).parent.absolute()
os.chdir(script_dir)

data_sets = [
            #  "Avana", 
            # "GeCKO", 
            # "RNAi_Ach", 
            # "RNAi_merged", 
            "RNAi_Nov_DEM"
            ]
iter = 750
trials = 300
k_list = list(range(3,20))

for data in data_sets:
    print("Starting batch run for: %s"%data)
    for k in k_list:
        a=sp.run(args=['python3','../run-nmf.py','-v', data,str(k), str(iter), str(trials)])
