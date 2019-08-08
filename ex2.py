help(round) #help on function

#functions
def least_difference(a=1,b=2,c=3): #example of default parameters
	"""
	This is a docstring
	
	Returns the smallest out of 3 given numbers.
	"""
	diff1 = abs(a-b)
	diff2 = abs(b-c)
	diff3 = abs(c-a)
	return min(diff1,diff2,diff3)
	
print(least_difference(1,10,100),least_difference(1,10,10),least_difference(1,2,3))
#,sep=" & ", end=". That's all!"
#seems like pythn 2.7 don't have such a function

help(least_difference)

def mult_by_five(x):
	return 5 * x

def call(fn,arg):
	return fn(arg)
	
print("You can pass a function as an argument - ")
print(call(mult_by_five,5))

#max function can also take in functions as arguments for comparison
def mod_5(x):
	return x % 5

print(max(100,51,14,key=mod_5)) #returns 14, the value which maximises the outcome of mod_5

round(3.1419,2) # round to 2 decimals
round(2123132,-1) # round to nearest 10