#use of the backslash character
print('It\'s name is Bob.')

#triple quote syntax
str = """This is
a
	special
string
"""

print(str)
print(str[0:2])

#backslash/escaping
str = "Pluto\'s\na planet. /\\"
print(str)

triple = """hello
world"""
hello = "hello\nworld"
print(triple == hello)

print(triple[3]) #strings are sequences
print([char+"_" for char in triple])

str.upper()
str.lower()
str.index('spe')
str.startswith('Thi')
str.endswith('Thi')

print(str.split(' '))
print("_".join(['hi','there']))

planet = "Pluto"
position = 9
"{}, you'll always be the {}th planet to me.".format(planet, position) #skip string formatting ie str(position)

pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
print(
"""{} weighs about {:.2} kilograms
	({:.3%} of Earth's mass).
	
	It is home to {:,} Plutonians.""".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)
)


## https://www.kaggle.com/colinmorris/strings-and-dictionaries