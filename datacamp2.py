"""iterables"""

word = 'Data'
it = iter(word)
print(next(it)) #"D"
print(*it)
print(*it) #nothing else to iterate

file = open("file.txt")
it = iter(file)
print(next(it)) #prints the first line
#note: word is an 'iterable' and it is an 'iterator'

"""More on Range"""
#iter and ranges: you can iter a range object
small_value = iter(range(3))
list(range(10,20))
sum(range(1,5))

"""enumerate and zip"""
#enumerate takes a list and gives u both element and index*
e = enumerate(["iron","hawk","thor"], start=0) #enumerate object, change start if you want the index to begin elsewhere
print(list(e)) # [(0, 'iron'), (1, 'hawk'), (2, 'thor')]
print('again: ',list(e)) # [(0, 'iron'), (1, 'hawk'), (2, 'thor')]
z = zip(["iron","hawk","thor"],['spark','barton','odinson'])
#print(*z) same as list(z) //splat operator
#print(list(z)) ##NOTE: After printing this the iterator finishes, rendering the below empty
for avengername, realname in z:
	print(avengername, realname)
z = zip(["iron","hawk","thor"],['spark','barton','odinson'])
print(list(z))
z = zip(["iron","hawk","thor"],['spark','barton','odinson'])
#unpack a zip with the splat operator
result1, result2 = zip(*z)
print(result1)
print(result2)

"""loading very large files"""
#import pandas as pd
#total=0
#for i in pd.read_csv("data.csv",chunksize=1000): #an iterable is created
#	total += sum(i['columnrequired'])
#print(total)

