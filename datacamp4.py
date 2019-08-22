"""Opening files"""
filename = ""
file = open(filename, mode='r')
text = file.read()
print(text)
file.close() #check w print(file.closed)

#using with statement/context manager
#creates a context with the file open
#once completed file is no longer opened
#this is binding in a context manager construct
#with open(filename, mode="r") as file:#

"""Using numpy for numerical data flat files"""
import numpy as np #for files that contain only numerical data
data = np.loadtxt('filename.txt', delimiter=',', skiprows=1, usecols=[0,2], dtype=str) #skiprows skip the header, usecols is which columns u want, dtype to force everything into strings
print(data)

"""MNIST"""
###this is very interesting
###mnist dataset loaded as a txt file and visualized
# Import package
import numpy as np

# Assign filename to variable: file
file = 'digits.csv'

# Load file as array: digits
digits = np.loadtxt(file, delimiter=",")

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()

"""Loading dataset containing multiple datatypes"""
#when u import a structured array (each row being a structured array of different types)
#also, need to use genfromtext
#data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
#or data = np.recfromcsv('titanic.csv', delimiter=',', names=True)
#data[i] for row and data['Column'] for column

"""Dataframes"""
#more useful for data scientists, modelling, splicing, groupbys, merge etc
data = pd.read_csv("filename.csv")
data.head()
convertedtonumpyarray = data.values
#data = pd.read_csv("digits.csv", header=None, nrows=5)
#data = pd.read_csv(file, sep="\t", comment='#', na_values=["Nothing"])

