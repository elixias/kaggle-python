x = True
print(type(x))

#part 1
a = 1
b = 2
print(a == b)
print(a <= b)
print(a >= b)
print(a != b)
print(a > b)
print(a < b)

#part 2
print(3.0 == 3) #true
print('3' == 3) #false

#instead of &&, ||, !=
#use and, or, not
print("Part 3")
print(True and False)
print(True or False)
print(True is not False)
print(not True)
print(True or True and False) #is True as 'and' takes precedence

if x == 0:
	print(x," x is zero")
elif x > 0:
	print(x," x is > zero")
elif x < 0:
	print(x," x is > negative")
else:
	print('oh my.')
	
#0 as well as any empty string/list/tuple etc returns False
print(bool(0))

#conditional expression
grade = 30
outcome = 'failed' if grade < 50 else 'passed'

#you can perform arithmetics on booleans..
#when ketchup onion and mustard are booleans..
ketchup = mustard = True
onion = False
print(ketchup+mustard+onion) # >>> 2
