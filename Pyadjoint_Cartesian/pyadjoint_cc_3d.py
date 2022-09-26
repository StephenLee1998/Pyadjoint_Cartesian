#! /usr/bin/env python
### This script is used to Cal cc adjoint sources for SPECFEM3D ###
### Jan 25 2022 , li_chao ### 

from operator import le
import os
import numpy
import obspy
from utils import load_config, validate_config
from bpfilter import bpfilter
from select_window import select_window
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()

    config = load_config(args.config_file)
    validate_config(config)

    cal_type = config["cal_type"]

    if cal_type == "BPfilter":
        bpfilter(config)
    
    if cal_type == "SelectWindow":
        select_window(config)



if __name__ == "__main__":
    main()
