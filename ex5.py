planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ')

#loops through char in string

#range
for i in range(5):
	print(i)

#while
i=0
while i < 10:
	print(i,end=' ')
	i+=1
print('\n')

squares = [i**2 for i in range(10)] #list comprehension
print(squares)

squares = [i**2 for i in range(10) if i > 5 ] #list comprehension WITH if condition
print(squares)

def has_lucky_number(nums):
	return any([num % 7 == 0 for num in nums]) # use of any*