#dictionaries
numbers = {'one':1,'two':2,'three':3}
print(numbers['one'])
#assigning
numbers['eleven'] = 11
print(numbers['eleven'])

print(numbers)

for i in numbers:
	print(i) #uses keys

#dictionary comprehension - similar to list comprehension
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
new_list = {p:"{} is the {}th planet in the solar system".format(p, planets.index(p)+1) for p in planets} #note: need to provide key and value using :
for i in new_list: # in keyword refers to the key, i.e: 'Saturn' in new_list >>> True
	print(new_list[i])

print(new_list.keys())
print(new_list.values())
print(new_list.items())

#iterating using key value pair instead of just key
for key, val in new_list.items():
	print("K:{} V:{}".format(key,val));

#the 'equivalent' for array is enumerate
for i, ele in enumerate(['this','is','a','list'):
	print("ind:{} ele:{}".format(i,ele))

