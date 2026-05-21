#there are two types of data structures in pandas
# 1. series , one dimensional array
# 2. dataframes , two dimensional array
from operator import index

import pandas as pd
l=[1,6,9]
s1 = pd.Series(l)
print(s1)
print(s1[1])# to check what element is present at index 1
##################
#s2 = pd.Series(l,index-['x','y','z'])
#print(s2)

#####################
#ages = pd.Series([22,35,51],name='Age')
#print(ages)

################
calories = {'day1':420,'day2':380,'day3':390}
s3 = pd.Series(calories)
print(s3)
print(s3['day1'])
##################
#dataframe
df = pd.DataFrame({'Name  =':['Ali','Zain','Wahab'],
                   'Age':[20,21,21],
                   'Gender':["M",'M','M']})
print(df)
#for accessing only one column
print(df['Age'])
# read a csv file
data = pd.read_csv('myfile.csv')
print(data)

# complete information
print(data.describe())
#if want to ready without header header=None
titanic = pd.read_csv('titanic.csv')
print(titanic)
# if only want to view data of last four
# tail display last records of data
t = titanic.tail(4)
print(t)
#display first records of data
t1= titanic.head(4)
print(t1)
#########################
print(titanic[['Age','Fare']])
c = titanic.iloc[3] # integer location picking column no 3
print(c)
