# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:22:06 2017

@author: kammittal
"""
# Python = object oriented programming

x= "hello world"
print(x.capitalize())
print("Hello world")

# variable assignment
var1 = list(range(3))
print(var1)
var1.append(10)

#Python keywords cannot be over-written, functions can be overwritten

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
