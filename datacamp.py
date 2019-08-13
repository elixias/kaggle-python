#functions
def square(a): #function header
	"""docstring"""
	return a**2

import pandas as pd
#df = pd.read_csv("", index_col="")
#df.loc / iloc / transpose / ix / 
#df[df.logical_and(...)]

#import matpplotlib.pyplot as plt
#plt.hist / plot / scatter(...,size=x)
#plt.xticks(xtickval, xticklbl)
#plt.xlabel/ylabel/title/show/grid(True)/xscale

#within the function if you use the global keyword, bcomes global var
"""LEGB: Local, Enclosing, Global, Builti"""
gvar = 10
def somefunction():
	global gvar
	gvar = 20
	
	nvar = 1
	def inner():
		nonlocal nvar # similar to global but in the enclosing fn
		nvar = 2
		print(nvar)
	inner()
	print(nvar)
print(gvar==20)
somefunction()

#inner functions
def raise_val(n):
	def inner(x):
		return x ** n
	return inner
square = raise_val(2) #this returns a function that ^2
cube = raise_val(3) #this returns a function that ^2
print(square(2),cube(2))

def somefunc(*args, hey='hm'): # for key value pairs use **kwargs
	for v in args:
		print("{}".format(v))
	print('hey value'+hey)
	#for k,v in kwargs:
	#	print("{} {}".format(k,v))
somefunc(1,1234,1234,hey='hey')

#lamba functions
nums = [48, 6, 9, 21, 1]
square_all = map(lambda num: num ** 2, nums)
print(list(square_all))
some_lamba_fn = lambda word:word*5
print(some_lamba_fn("hi"))
#use of filter
#  result = filter(lambda member:len(member)>6, fellowship)

def sqrt(x):
	if x < 0:
		raise ValueError("x must be non neg") #raise Exception()
	try:										#try, except
		return x ** 0.5
	except TypeError:
		print("x must be int or flat")	