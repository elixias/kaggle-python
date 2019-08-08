spam = 0
print(spam)

spam = spam + 4

if spam > 0:
	print("I don't want spam.")

song = "Spam " * spam
print(song)

print(type(spam))
if(type(spam) is int):
	print("It's an INT")
print(type(song))	
if(type(song) is int):
	print("It's an INT")

print(type(type(spam))) #it's a type

##Arithmetic Operators
a = 5
b = 3

print(a+b)
print(a-b)
print(a*b)
print(float(a)/b) #does not do true division unless one of them is a float
print(a//b)
print(a%b)
print(a**b)
print(-a)
print(~a)

print(min(1,2,3,4,5))
print(max(1,2,3,4,5))
print(abs(-32))
print(float(3))
print(int('3'))

##easily swap values between 2 variables
a, b = b, a #tuples

print(b)
print(a)
