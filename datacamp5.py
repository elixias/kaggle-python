#excel xls spreadsheets
import pandas as pd
data=pd.ExcelFile('urbanpop.xlsx')
print(data.sheet_names)
df1=data.parse('tab_name') # or df1=data.parse(0) #sheet index
#customizing
#df=data.parse('tab_name',skiprows=[0],names=['rename'], usecols=[0]) #names arg is to set the column names

#OS
import os
os.listdir(os.getcwd())

#Pickle files
# Import pickle package
import pickle
with open('data.pkl', mode="rb") as file: # Open pickle file and load data: d
    d = pickle.load(file)
print(d)# Print d
print(type(d))# Print datatype of d

#SAS statistical analysis system, Stata = statistics+data
import pandas as pd
from sas7bdat import SAS7BDAT #or if SAS7BCAT
with SAS7BDAT('urbanpop.sas7bdat') as file:
	df_sas = file.to_data_frame()

data = pd.read_stata('urbanpop.dta') #for stata files

#for HDF5 files Hierachical Data Format
import h5py
data = h5py.File('','r')
#data.keys() #meta, quality, strain
#data['_']['_'].value

#matlab matrix lab (.mat)
#scipy.io.save_mat/load_mat
import scipy.io
mat = scipy.io.loadmat('filename.mat')
print(mat['x'])

#sql, dataacademy uses sqlalchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///northwind.sqlite')
table_names = engine.table_names()
print(table_names)
con = engine.connect() #connect to the engine
rs = con.execute('SELECT * FROM orders') #sqlalchemy results object
df = pd.DataFrame(rs.fetchall())
#df = pd.DataFrame(rs.fetchmany(size=5))
df.columns = rs.keys() #keys have to be explicitly set
con.close() #close connection

"""to prevent forgetting to close something try using context manager"""
#with engine.connect() as con:
#	...

"""an easier method to do everything in one line"""
df = pd.read_sql_query('SELECT * FROM ORDERS', engine)

