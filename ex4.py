#lists in python, array
primes = [2,3,5,7]
#and can contain different values as well as lists
a_list = ["this",[1,"hello",3],"a list!"]
print(a_list)

#zero-based
print(primes[0])
print(primes[-1]) #first element from end (i.e: last) of list
print(primes[1:3]) #primes[1] to primes[2]
print(primes[1:]) #from index[1] to end of list
print(primes[:4]) #from index[0] to index[4]
print(primes[-3:]) #last 3 elements
print(primes[:-1]) #except last

#you can assign values to them too
primes[:3] = [1,1,1] #overwrite first 3 elements
print(primes)

print("Length of list",len(primes))
print(sorted(primes))
print("Sum",sum(primes))
print("Max",max(primes))
print("Min",min(primes))

x = 12
#attribues x.imag as in imaginary
#method x.bit_length()

#list related methods
primes.append(11)

#addition also lets you append
primes = primes + [13,17]
del(primes[-2:-1]) #removing elements
print('New Prime',primes)

new_prime_list = primes
###changing elements in new_prime_list also changes primes. variable is a reference to address.
new_prime_list += [13]
print(primes)

#thus to copy a list
new_prime_list = list(primes)
# or
new_prime_list = primes[x]

primes.pop()
if 7 in primes: #you need to check first else error if not found
	print(primes.index(7))

#tuples
another_prime = 2,3,5,7,9,11 #cannot be modified #you can also use ()
