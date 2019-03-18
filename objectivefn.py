#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:42:41 2018

@author: reggiewirya
"""
from math import log, exp
import torch

def ObjectiveFunction(w,x,y):
    i = 0
    logsum = 0
    while i<=9999:
        logsum = logsum + log(1 + exp(-y[i] * torch.mm(torch.t(w),torch.t(x[i:i+1,:]))) )
        i = i + 1
    objfunc = logsum/10000
    return objfunc
