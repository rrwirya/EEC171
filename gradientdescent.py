'''
Homework 3a) and 3b) ~ Gradient Descent
Reggie R Wirya - 913184123
'''

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
    #return torch.FloatTensor(line)
    return torch.DoubleTensor(line)
    #return torch.IntTensor(line)

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
    #return torch.FloatTensor(line)
    return torch.DoubleTensor(line)
    #return torch.IntTensor(line)

def gradE_in(w,x,y):
    '''
    Calculate the gradient of the in-sample Error (E_in)
    Input:  w = weight [780x1 FloatTensor]
            x = xList
            y = yList
    Output: result of gradient [780x1 FloatTensor] with the determined input variables
    ''' 
    i = 0
    grad = torch.t(torch.DoubleTensor(780,1).zero_())
    while i<=9999:
        grad = grad + torch.round(-y[i] * x[i]/(1 + exp(y[i] * torch.mm(torch.t(w),torch.t(x[i:i+1,:])))) )
        print(y[i] * torch.mm(torch.t(w),torch.t(x[i:i+1,:])))
        i = i + 1
    grad = torch.round(-(grad/10000))
    return torch.t(grad)

def minW(x,y):
    '''
    Calculates the optimum w vector through iteration (in this case only done once)
    Input:  x = xList [DoubleTensor size 10000]
            y = yList [DoubleTensor size 10000]
    Output: optimum w returned in DoubleTensor [780x1]
    '''
    w = torch.DoubleTensor(780,1).zero_()
    prev_w = w
    i = 0
    step = 0.00001
    
    while i<500:
        grad = gradE_in(w,x,y)  # [780x1 FloatTensor]
        w = w + step * grad     # [780x1 FloatTensor]
        prev_w = w
        i = i + 1
    '''
    grad = gradE_in(w,x,y)
    w = w + step * grad
    '''
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

def ObjectiveFunction(w,x,y):
    # Measures the objective function for a certain w parameter based on the formula provided on the hw sheet
    i = 0
    logsum = 0
    while i<=9999:
        logsum = logsum + log(1 + exp(-y[i] * torch.mm(torch.t(w),torch.t(x[i:i+1,:]))) )
        i = i + 1
    objfunc = logsum/10000
    return objfunc

# test variables
x = TestFileX()
y = TestFileY()
testw = torch.DoubleTensor(780,1).zero_()

# Program
w = minW(x,y)
err = ErrorCount(w,x,y)

'''
95.8% accuracy learning model = 420 mismatch (iteration = 1, step size = 0.1)
stopped at 1 iteration due to error
rounding used -> more inaccuracy
2nd iteration = exp(15811) -> OverflowError: math range error
objectivefunction = 0.2578555527887662

98.12% accuracy learning model = 188 mismatch (iteration = 100, step size = 0.0001)
rounding used -> more inaccuracy
objectivefunction = 0.07030177807499005

98.12% accuracy learning model = 188 mismatch (iteration = 500, step size = 0.0001)
rounding used -> more inaccuracy
objectivefunction = 0.07030177807499005

98.31% accuracy learning model = 169 mismatch (iteration = 1000, step size = 0.00001)
rounding used -> more inaccuracy
objectivefunction = 0.0587785195519154
'''
