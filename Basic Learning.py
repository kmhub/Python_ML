# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:22:06 2017

@author: kammittal
"""
# Python = object oriented programming
#Python keywords cannot be over-written, functions can be overwritten
#white space is significant and implies the various control flow structures.
#strings don't support item assignment, thus they are immutable
#Numbers are smaller than strings when using comparison operator <,>;  string case matters for comparison purposes
#Tuples() are immutable, fixed length vs. Lists [] are the mutable variable length
#Pandas uses NaN to cover missing values

x= "hello world"
print(x.capitalize())
print("Hello world")

# variable assignment
var1 = list(range(3))
print(var1)
var1.append(10)

# Built-in functions vs functions atteched to objects(also called Methods)
#Function
var1= 'foo'
print(len(var1))

#Method
print(var1.capitalize())

#Attributes
print(var1.len())

#Importing Modules. Modules can be user defined( scripts ending in py)
import keyword as kw # (give module short alias name)
kw.kwlist
from keyword import kwlist # (Load all module objects into local namespace…not recommended)

#string data type; strngs are immutable
var1[2] = "c" # returns error

#If you tried a[1] = 'z' to make "dog" into "dzg", you would get the error: #TypeError: 'str' object does not support item assignment" 
#because strings don't support item assignment, thus they are immutable.

var1 = 'Foo'
for i in var1:
    print(i)

var1[0:2]
print(var1.count("o"))

#placeholder for strings
print("Hello, the name is {name}".format(name="John"))
print("Hello, the name is {fname} {lname}".format(fname="John",lname="Junior"))
print("Hello, the name is {0} {1}".format("John","Junior"))
print("Hello, the name is {} {}".format("John","Junior"))
print("Hello, the name is {0:_<6}{1}".format("John","Junior"))

#prompt user for a input
p = input("what is your name")
print(p)

#integer, float
x=43
y= 9
x/y

abs(x)

# math module
import math
math.sqrt(45)

#type casting
str(8)
int("  45")
float("82348 ")
bool(x=43)
datetime.strptime() # string to date object
datetime.strftime()

tuple()



#null values = None
xyz=None
print(xyz)

#module datetime
from datetime import date, time, datetime
print(date(2014,7,23))

dt2 = datetime.strptime("2014/07/11","%Y/%m/%d") # string to date object
print(dt2)
dt2.strftime("%Y-%m") # date to string object


#control flow structures
# If else
if 2 < 3:
    print("This is right" )
else:
    print("This is wrong")
# On one Line
print("This is right") if 2 < 3 else "wrong"
#nesting
x = -5
if x > 0:
    print("greater than zero")
elif x == 0:
    print("zero")
elif x < 0:
    print("less than zero")
    
#for loop
for i in "John junior":
    if i.upper() != "J":
        print(i)    
        
# While loop
x = 7
while x > 0:
    print(x)
    x = x - 1
    
#exception handling - try except
try:
    int("1 2  3")
except ValueError:
    print("1 2  3")
        
a = 100
b = "Jon"
try:
    print(a + b)
except Exception as e:
    print("check...\n", e)

# Data structures
# Strings, floats, and integers are data structures
#Tuple -- Immutable, fixed length sequence
#Lists -- Mutable sequence of objects
#Dicts (dictionary) -- List of objects with a "key"
#Sets -- Unique collection of objects

# Tuple ( seuqence of characters,numbers,etc. - immutable)
tup1 = 1,2,3,4,5,6
tup2 = (1,2,3,4,5,6)
tup1 == tup2
# Immutable, raises TypeError: 'tuple' object does not support item assignment
tup1[1] = 10

b = tuple("long text written")
print(b)

# Nesting of tuples
nest1 = (("the","first"),("the","second"))
nest1[0]
nest1[0][0]
nest1[0][0][2]

#Unpacking Tuples -  "unpack" themselves upon assignment
a, b = nest1
print(a)

(a1,a2),(b1,b2) = nest1
print(a1)

# lists - Mutable, variable-length collection of objects
list123 = [3,4,5,6]
# mutable
list123[2] = 1
list123.append(36)
print(list123)

# List functions
list123 = ["a","b","c","x"]
list123.append("d")
list123.insert(1,"a.5")
list123.remove("a.5")
list123.append(["d","e","f"]) # nested list

list123.extend(["d","e","f"]) # works as expected

# indexing
seq1 = list(range(1,11))
seq1[0:2:2]

#Dicts - maps values to objects, mutable
d1 = dict(first="a", second="b", last="c")
d2 = dict(zip(['first', 'second', 'last'], ["a", "b", "c"]))
d3 = {'first': "a", 'second': "b", 'last': "c"}
d4 = dict([('second', "b"), ('first', "a"), ('last', "c")])
d1==d2
d1["first"]
# Remove element
del d1["first"]
# Update element
d1.update({'second':'changed'})

#comprehension
list1 = range(10)
list2 = []
for i in list1:
    list2.append(i + 1)
list2

list2 = [i+1 for i in list1]
list2 = [i+1 if i > 4 else i for i in list1]

#functions
def mult_by_3(x,w=1):
    output = x*w*3
    return output

mult_by_3(4,5)

#global variable function
# Global var function
def mult_by_3(x,w=1):
    global output
    output = x*w*3
    return output

mult_by_3(4)
output
#return multiple values
def multi_func(a,b,c):
    return (a + 1,b + 3,c - 3)

a1, b1, c1 = multi_func(1,1,1)

#lambda - one line function expressions
lam_func = lambda x1,x2: x1 + x2
lam_func1 = lambda x1,x2: (x1 + x2,x1 - x2)
lam_func(3,4)
lam_func1(3,4) 

# working directory
import os
os.getcwd()
# Set Working Directory
os.chdir(r"C:\\Users\\kammittal\\Documents\\GitHub\\Python_ML")

#module numpy
import numpy as np

ar1 = np.array(range(10))
ar2 = np.array([[1,2,3],[4,5,6]])
ar1.ndim
ar2.ndim
ar2.shape
ar2.dtype
ar2.size

np.ones((2,4)) # create an array of all ones
np.zeros((2,4)) # create an array of all zeros
np.identity(5) # create an identity matrix

# change array dtype
ar3 = np.array([1,2,3,4,5,6],"float")
ar3.dtype
# change dtype
ar3 = ar3.astype("int")

#reshaping array
line = np.arange(1,4*4*4+1)
square= line.reshape(4,16)
cube = line.reshape(4,4,4)

cube[0,2,2]

np.random.seed(123)
norm_array = np.random.randn(10).reshape(5,2)
truevalues = norm_array > 1
norm_array[truevalues]

norm_array[norm_array > 0] = 0

# Vectorized calculations
norm_array + 1

np.mean(norm_array)
np.mean(norm_array,0)
np.mean(norm_array,1)
np.var(norm_array) # variance ~ average of the squared deviations from the mean

# module np.random 
np.random.rand(4,4)
# normal(mean=0,var=1)
np.random.randn(2,2,2)
4*np.random.randn(2,2,2) + 10
# binomial
np.random.binomial(n=100,p=0.2,size=50)

#module pandas
import pandas as pd
np.random.seed(123)
data = {'custid': list(range(1,11)), 'year': list(range(2010,2015))*2, 'spent': np.random.randint(10,50,10)}
Frame = pd.DataFrame(data)
Frame

#Assign Index
Frame1 = pd.DataFrame(data, index=["one","two","three","four","five", "six","seven","eight","nine","ten"])
# or shortcut
Frame1.index=["one","two","three","four","five", "six","seven","eight","nine","ten"]

#Change Columns
Frame.columns = ["custid","date","mon_spent"]

#Import CSV to pandas DF
states = pd.read_csv(r"states.csv")

# Check data after import
states.head(n=5)
states.tail(n=5)
states.describe()
# Number of Rows
len(states)
#List Columns
states.columns
# List row IDS
states.index

# Select Column
states['State']
states[['State','Population']]
# By attribute
states.State
states[0:2]
states[states.Population < 1000]
states[states < 1000]

#Index methods
# select by positional index, # end exclusive
states.iloc[0:3,0:4] 

# select by label index, # end inclusive
states.loc[0:15,"Population":"Life Exp"]

# Add and Delete Column
states["Districts"] = 10
states.head()

del states["Districts"]
states.head()

# Delete Row/Add Row
states.drop(states.index[0])
states.drop(states.index[-1])
states1 = states[0:10]
states2 = states[10:25]
states3 = states[25:]
states1.append([states2,states3])

# Reindex
states.sub = states[states.Population < 1000]
states.sub = states.sub.reset_index()
del states.sub["index"]
states.sub

#Missing values
States_Nan = states
States_Nan.Population[states.Population < 1000] = np.NaN
#drop missing data
States_Nan.dropna()
states.fillna(0)
states.fillna(method="ffill")

# isnull
States_Nan.isnull()
States_Nan.notnull()

# Functions on DataFrames

states.mean()
#Whats the Difference between these two?
states.mean()["Population"]
states["Population"].mean()

states.sort_values("Population", ascending=True)
states.sort_values(by=["Illiteracy","Population"], ascending=[True,False])

states["Illiteracy"].unique()

states.drop_duplicates()

# Merging dataset
states = pd.read_csv(r"states.csv")
states2 = pd.read_csv(r"states_extra.csv")
# Merge these together
pd.merge(states,states2, how="left", left_on="State",right_on="ST")


# reshaping a dataframe
state_sub = states.iloc[0:3,1:3].copy()
state_sub.index = states.State[0:3]
state_sub.stack()
state_sub.stack().unstack()
# specify which dimension to unstack
state_sub.stack().unstack(0)
state_sub.stack().unstack(1)

df = pd.DataFrame({"id": ("a","a","b","b","c","c"), "time": (1,2,1,2,1,2), "meas":range(6)})
df.pivot("id","time","meas")

# Duplicates
df_dup = df.append(df.iloc[1])
df_dup.duplicated()
df_dup.drop_duplicates()

#Bins/ categories
states.Population.describe()
bins = [0,1000,3000,5000,25000]
pd.cut(states.Population, bins)
# With Labels
states['Population_bins']= pd.cut(states.Population, bins, labels=["small","med","large","huge"])


# Group By
grp = states.groupby("Population_bins")
grp.mean()
grp.var()
# group by one variable
states.Income.groupby(states.Population_bins).mean()
# dynamic group by one variable and index
states.Illiteracy.groupby(states.Income < 5000).mean()
# Eliminate the index
grp = states.groupby(states.Income < 5000,as_index=False)
grp.mean()

# Transform
grp = states.groupby(states.Income < 5000,as_index=False)["Illiteracy"]
grp.transform(np.mean)

# Cross Tabs/ Frequency table
pd.crosstab(states.Income < 5000,states.Population_bins)
pd.crosstab(states.Income < 5000,states.Population_bins,margins=True)

#Create the table
l1 = pd.crosstab(states.Income < 5000,states.Population_bins)
# Applying Functions
l1.apply(lambda x: x/x.sum()*100,axis=0)
l1.apply(lambda x: x/x.sum()*100,axis=1)

# module Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Line plots
np.random.seed(123)
x = np.arange(1,7)
y = np.random.randn(6)
plt.plot(x,y)
# Red dots
plt.plot(x,y,"ro")

fig = plt.figure()
plt.title("First Plot")
plt.xlim(0,7)
plt.plot(x,y,"--")
plt.plot(x,y,"ro")
plt.ylabel("Mood")
plt.xlabel("DayofWeek")

# Histogram
np.random.seed(123)
b = np.random.binomial(100,0.2,1000)
plt.figure()
plt.title("Binomial")
plt.hist(b)

plt.grid(True)
plt.hist(b,bins=20,  orientation = "vertical", color="b",alpha=0.75)

# Save figure
plt.savefig("plot.png")

# Import Dataframe
states = pd.read_csv(r"states.csv")
states["Income_bins"] = pd.cut(states.Income,3,labels=["low","med","high"])
# Histograms
states.Income.plot(kind="hist")
states.Income.hist(color='k', alpha=0.5, bins=50)