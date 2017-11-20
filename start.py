#!/usr/bin/env python3.6

import subprocess
import signal
import os
import time

for i in range(3):
    devnull = open('/dev/null', 'w')
    p = subprocess.Popen(["./crypto_analysis.py"], stdout=devnull, shell=False)
    
    print("process {} started!".format(p.pid))

    time.sleep(3600)

    # Get the process id
    os.kill(p.pid, signal.SIGINT)

    if not p.poll():
        print("Process correctly halted")
