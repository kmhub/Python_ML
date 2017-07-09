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

# Tuple
tup1 = 1,2,3,4,5,6
tup2 = (1,2,3,4,5,6)
tup1 == tup2
# Immutable, raises TypeError: 'tuple' object does not support item assignment
tup1[1] = 10


