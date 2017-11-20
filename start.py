#!/usr/bin/env python3.6

import subprocess
import signal
import os
import time

while True:
    devnull = open('/dev/null', 'w')
    p = subprocess.Popen(["./analysis.py"], stdout=devnull, shell=False)
    
    print("process {} started!".format(p.pid))

    time.sleep(300)

    # Get the process id
    os.kill(p.pid, signal.SIGINT)

    if not p.poll():
        print("Process correctly halted")
