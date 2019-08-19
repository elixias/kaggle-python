#list comprehensions
#previously went through that you can make list comprehensions using existing series
#but, you can use it on any iterable object

result = [ num for num in range(11) ]
print(*result)

#creating a list of tuples with all integer combinations from 2 series
pair = [ (num1,num2) for num1 in range(0,2) for num2 in range(6,8) ]
print(pair)

#you can specify the list comprehension as the output expression and use list comprehension to iterate the number of cycles*
matrix = [ [col for col in range(0,5)] for row in range(0,5) ]

#you can include conditional in comprehension or in the output expression

print([ num ** 2 for num in range (0,10) if num % 2 == 0 ]) #outputs only when conditionals are met, placed in predicate expression
#[0, 4, 16, 36, 64]
print([ num ** 2 if num % 2 == 0 else 0 for num in range (0,10)]) #AND/OR outputs based on conditionals, the else is required
#[0, 0, 4, 0, 16, 0, 36, 0, 64, 0]
#new_fellowship = [member if len(member)>6 else "" for member in fellowship]
# if _conditional_ else ___

#dict comprehensions are similar except the outputs are a pair of key value separated by :

"""generators"""
#replace [] with (). does not store in memory. however when u loop through them the elements are generated.
#you can use print, list(<generator object>) or next(<generator obj>) 
#this is called lazy evaluation <- delayed until needed
even = (num for num in range (0,10) if num%2==0)
print(list(even))

#generator function
#uses the keyword "yield"
def num_seq(n):
	i = 0
	while i < n:
		yield i ## instead of return
		i += 1
res = num_seq(5)
print(res) #returns generator object
print(list(res))

"""creating dict with a zipped list"""
#dict(zip([1,2,3,4],[5,6,7,8]))
#{1: 5, 2: 6, 3: 7, 4: 8}

#when you have a list of dictionaries with keys(header column) and values,
#use pandas.DataFrame(<dict>) to convert into a df and use df.head()

#with open('world_dev_ind.csv') as file:
#with is context manager
#file.readline()

#you can use function generators to read a large file which yield the result when next() is called
#with context/file already using a generator no need to redefine a generator for this

# just an example as illustration from datacamp below:
# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))