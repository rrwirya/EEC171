'''
Homework 3c) and 3d) ~ Stochastic Gradient Descent
Reggie R Wirya - 913184123
'''

import random
from math import exp
from math import log
from math import floor
from numpy import transpose
import torch

def TestFileX():
    '''
    To read data from mnist_X_train.dat, returns the list of lists
    N = 10000 -> index until 9999
    '''
    file = open("mnist_X_train.dat")
    lines = str(file.readlines())
    lines = lines.split("\\n")
    
    line = []   
    i = 0       
    while i<(len(lines)-1):
        line.append(lines[i])
        line[i] = line[i].replace('[',' ').replace(']',' ').replace("'",' ')
        line[i] = [int(s) for s in line[i].split() if s.isdigit()]
        i = i + 1  
    return torch.DoubleTensor(line)

def TestFileY():
    '''
    To read data from mnist_y_train.dat, returns the list of lists
    N = 10000 -> index until 9999
    '''
    file = open("mnist_y_train.dat")
    lines = str(file.readlines())
    lines = lines.split("\\n")
    
    line = []
    i = 0
    while i<(len(lines)-1):
        line.append(lines[i])
        line[i] = line[i].replace('[',' ').replace(']',' ').replace("'",' ').replace(",",' ')
        line[i] = int(line[i])
        i = i + 1
    return torch.DoubleTensor(line)

def RandomIndex(n):
    '''
    Input:  n = length of intended Batch size
    Output: n random indexes in a list for the x and y torch DoubleTensor arrays
    '''
    return random.sample(range(0,9999),n)

def RandX(x,rand):
    '''
    Input:  x = DoubleTensor of size [10000x780]
            rand = list of indexes set in random from RandomIndex function
    Output: new_x = DoubleTensor of size [len(rand)x780]
    '''
    new_x = torch.DoubleTensor(len(rand),780)
    i = 0
    while i<len(rand):
        new_x[i] = x[rand[i]]
        i = i + 1
    return new_x

def RandY(y,rand):
    '''
    Input:  y = DoubleTensor of size[10000]
            rand = list of indexes set in random from RandomIndex function
    Output: new_y = DoubleTensor of size [len(rand)]
    '''
    new_y = torch.DoubleTensor(len(rand),1)
    i = 0
    while i<len(rand):
        new_y[i] = y[rand[i]]
        i = i + 1
    return new_y

def minW(xrand,yrand,rand):
    '''
    Calculates the optimum w vector through iteration (in this case only done once)
    Input:  x = xList [DoubleTensor size 10000]
            y = yList [DoubleTensor size 10000]
    Output: optimum w returned in DoubleTensor [780x1]
    '''
    batchsize = len(rand)+1
    w = torch.DoubleTensor(780,1).zero_()
    grad = torch.DoubleTensor(780,1).zero_()
    idx = 0
    t = 1
    const = -3
    while t<batchsize:
        xprocess = xrand[idx:idx+1,:]
        yprocess = yrand[idx]
        #print(torch.mm(torch.t(w),torch.t(xprocess)))
        exp_part = torch.round(yprocess * torch.mm(torch.t(w),torch.t(xprocess)))
        print(exp_part)
        #print('lambda')
        grad = torch.round(-yprocess * xprocess/(1 + exp(exp_part)) )
        #print(grad)
        w = torch.round(w - (t**const)*torch.t(grad))
        idx = idx + 1
        t = t + 1
    return w

def ErrorCount(w,x,y):
    '''
    Counts mismatches by computing the total amount of difference between y[i] and wTx[i] on all i until 9999
    Input:  w = weight vector
            x = xList
            y = yList
    Output: returns total number of errors. Divide by 100 to get the error percentage
    '''
    w_x = torch.mm(torch.t(w),torch.t(x))
    transpose_wx = torch.t(w_x)
    result = []
    z = 0
    while z<10000:
        if(float(transpose_wx[z])>=0):
            result.append(1)
        else:
            result.append(-1)
        z = z + 1
    i = 0
    error = 0
    while i<10000:
        if(y[i] != result[i]):
            error = error + 1
        i = i + 1
    return error

# test variables
rand = RandomIndex(100)
xrand = RandX(TestFileX(),rand)
yrand = RandY(TestFileY(),rand)
x = TestFileX()
y = TestFileY()

# Program
w = torch.DoubleTensor(780,1).zero_()
