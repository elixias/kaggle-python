import math as mt #importing an external *module* caled math # as keyword renames the module
#import math as * #if you want to avoid typing math in front of every namespace i.e: print(pi)
#import math as pi, log #for specific modules as importing * from multiple modules lead to unexpected results

###submodules
#import numpy.random as randint #modules can contain modules ie submodules

# a module is a collection of variables, or, namespace
print(dir(math)) #prints all the 'names' available in the module

print(math.pi)
print(math.log(32,2))

help(math) #use the help method on the math module itself to see the docstrings for all functions and variable declarations

#exploring modules
#type()
#dir()
#help()

#numpy allows you to overload operators
#by allowing arithmetics and logical operators to be used on arrays
