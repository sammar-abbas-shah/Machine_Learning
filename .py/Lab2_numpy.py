import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import metrics
# This is a list
#l1 = [2,3,-5,0.777,1.333,'c++','java']
l1 = [2,4,6,8,'sammar']
print(l1)
#arr = np.array(l1,dtype=float)
#print(arr) # list is comma separated and array is space separated
arr = np.array(l1)
print(arr)
l2 = [[2,3],[4,5],[6,7]]
a2 = np.array(l2)
print(a2)
#l3 = [45,134,300]
#a3 = np.asarray(l3,dtype = np.int8)
#print(a3)
# 300 takes more than 7 bits so result is not saved in array
#l4 = [45,30,-10]
#a4 = np.asarray(l4,dtype=np.uint32)
#print(a4)
#unit can not save a negative number so
#arr5 = np.asarray([5,7,9,10],dtype=np.float_)
a7 = np.arange(7)
print(a7)#[0,1,2,3,4,5,6]
#if we want to declare start and end
a8 = np.arange(3,12)
print(a8) #[ 3  4  5  6  7  8  9 10 11]
# if we want to also tell how much increment
a9 = [np.arange(6,19,3)]
print(a9) #[array([ 6,  9, 12, 15, 18])]
a10= np.zeros((5,9))
print(a10)
a11 = np.ones((10,6),dtype=int)
print(a11)
a12 = np.array([3,1,7,8,9,4,1,6]) # dividing array into two
a13 = a12[:4]
a14 = a12[4:]
#now doubling the first array
a14 = a13*2
