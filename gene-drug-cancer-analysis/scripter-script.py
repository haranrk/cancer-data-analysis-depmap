from pathlib import Path as pth
import os
main_script_dir = pth(__file__).parent.absolute()
def scripter(py_command, loc):
    script_file = open("run.sh", 'w')
    script_file.write('#!/bin/sh\n')
    script_file.write('#$ -S /bin/sh\n')
    script_file.write('#$ -o %s' % loc)
    script_file.write('#$ -e %s' % loc)
    script_file.write('#$ -M haranrk@hgc.jp\n')
    script_file.write('#$ -m bae\n')
    script_file.write('#$ -N depmap\n')
    script_file.write('#$ -cwd -j y -l ljob -l s_vmem=12G -l mem_req=12G\n')
    script_file.write('echo "START: " `date  "+%Y%m%d-%H%M%S"`\n')
    script_file.write('start_time=`date +%s`\n')
    script_file.write('module load python/3.6.4\n')
    script_file.write(py_command)
    script_file.write('echo "DONE: " `date  "+%Y%m%d-%H%M%S"`\n')
    script_file.write('end_time=`date +%s`\n')
    script_file.write('duration=$((end_time - start_time))\n')
    script_file.write('echo "TIME: " $duration\n')
    script_file.close()

k_list = [3, 4, 5]
datasets = ["Avana",  "RNAi-Nov_DEM"]

for data in datasets:
    for k in k_list:
        os.chdir(main_script_dir.absolute())
        slave_script_dir = main_script_dir / "shirokane_runs" / data / str(k)
        slave_script_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(slave_script_dir.absolute())
        scripter("python %s/run-nmf.py %s %s\n" % (main_script_dir.absolute(), data, k), slave_script_dir)
        print("Created %s" % slave_script_dir)
