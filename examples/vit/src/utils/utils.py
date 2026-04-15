# This file contains code derived from the NOLA project:
#   https://github.com/UCDvision/NOLA
#
# Copyright (c) 2023 UCDvision
# Copyright (c) 2026 Bangguo Ye, Yuanwei Zhang, Xiaoqun Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import time
import sys

def initLogging(logFilename):
    """
    Initialize logging configuration to output logs to both the console and the specified file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s-%(levelname)s] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(logFilename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

def init(taskfolder, logfile):
    """
    Initialize task environment: create task directory, configure logging, 
    and return a time-based checkpoint path.
    """
    os.makedirs(taskfolder, exist_ok=True)
    initLogging(logfile)
    
    datafmt = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(taskfolder, f"{datafmt}.pt")
    
    return ckpt_path
