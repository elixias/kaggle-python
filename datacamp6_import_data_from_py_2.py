"""importing files from an online resource"""
from urllib.request import urlretrieve, urlopen, Request
url="https://www.simplesite.com/"

#method1, retrieve and write
urlretrieve(url,"file_to_write_to")

#method2, getting html response
"""response = urlopen(Request(url))
html = response.read()
print(html)
response.close()"""

#method3, html response w requests package
import requests
r = requests.get(url)
print(r.text)

#to load the online csv file directly, just use read_csv as normal
#df = pd.read_csv(url,sep=';')

#xl = pd.read_excel(url,sheetname=None) #None means all sheets
#xl.keys() #xl is dict object

"""Beautiful Soup"""
from bs4 import BeautifulSoup
soup = BeautifulSoup(r.text)
print(soup.prettify())
print(soup.title)
print(soup.get_text())
#find_all('a') <- [link.get('href') for link in find_all('a')]

"""Connecting to JSON"""
import json
with open('file.json','r') as json_file:
	json_data = json.load(json_file)
"""connecting to APIs to receive JSON"""
new_r = request.get(url)
json_data = r.json()