#!/usr/bin/env python

import subprocess
import signal
import os
import time

while True:
    devnull = open('/dev/null', 'w')
    p = subprocess.Popen(["./active_analysis.py"], stdout=devnull, shell=False)
    
    print("process {} started!".format(p.pid))

    time.sleep(600)

    # Get the process id
    os.kill(p.pid, signal.SIGINT)
    os.system("killall python3.6")

    time.sleep(30)
