#######################
# Import packages
#######################
import numpy as np
import seaborn as sns

import pandas as pd
# Set pandas settings to show all data when using .head(), .columns etc.
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import itertools # Pour créer des iterateurs

######################
# PLOTTING
######################
import matplotlib.pyplot as plt
%matplotlib inline

# BOKEH 
from bokeh.plotting import figure # Importation de la classe figure qui permet de créer un graphique bokeh.
from bokeh.io import  push_notebook, output_notebook, show
output_notebook() # permet d'afficher tous les futurs graphiques dans l'output d'une cellule jupyter. Si cette instruction n'est pas lancée, la figure s'affichera dans un nouvel onglet.
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge
from bokeh.models.tools import HoverTool

#######################
#Load data
#######################
df = pd.read_csv('ASRS_20y_data.csv', low_memory=False)

print("The raw dataset has a total of", len(df), "entries.",
     "\n/!\ These include unmanned-aircraft-related entries; yet the latter are insignificantly low in number (see below)")
 
#######################
# Remove UAS-related data (Unmanned Aircraft Systems)
#######################

#Find how many UAS I have in my Dataset
features_UAS_all = [
###################################
### FEATURES RELATED TO DRONES  ###
###################################
 'Latitude / Longitude (UAS)',
    'Operating Under Waivers / Exemptions / Authorizations (UAS)',
  'Waivers / Exemptions / Authorizations (UAS)',
  'Airworthiness Certification (UAS)', # Only 'Standard' value
  'Weight Category (UAS)',
  'Configuration (UAS)',
  'Flight Operated As (UAS)',
  'Flight Operated with Visual Observer (UAS)', # BVLOS (Beyond Visual Line of Sight) is a term relating to the operation of UAVs (unmanned aerial vehicles) and drones at distances outside the normal visible range of the pilot.
 'Control Mode (UAS)',
 'Flying In / Near / Over (UAS)',
 'Passenger Capable (UAS)', # There are only NaNs and 'N'
 'Type (UAS)',
 'Number of UAS Being Controlled (UAS)',
 'UAS Communication Breakdown',    
 'Airspace Authorization Provider (UAS)', # only NaN's in this feature --> drop ?

    
##################################################
### Additional entries for the above variables ###
##################################################
 'Airspace Authorization Provider (UAS).1',
 'Operating Under Waivers / Exemptions / Authorizations (UAS).1',
 'Waivers / Exemptions / Authorizations (UAS).1',
 'Airworthiness Certification (UAS).1',
 'Weight Category (UAS).1',
 'Configuration (UAS).1',
 'Flight Operated As (UAS).1',
 'Flight Operated with Visual Observer (UAS).1',
 'Control Mode (UAS).1',
 'Flying In / Near / Over (UAS).1',
 'Passenger Capable (UAS).1',
 'Type (UAS).1',
 'Number of UAS Being Controlled (UAS).1',    
##################################################
]

# Search and gather the rows that contain non-nan values for any of the UAS features into dataframes

files = [] # instantiate empty list

for feat in features_UAS_all:
    
    # append pandas DataFrames to 'files' --> files becomes list of df's.
    files.append(df.loc[df[feat].isna() == False])

# concatenate df's into 1 df:
df_UAS = pd.concat(files)

# Drop the duplicates; there definitely exist many, because we searched column-by-column
df_UAS = df_UAS.drop_duplicates(keep= 'first')

# Drop the UAS entries
df = df.drop(df_UAS.index, axis =0)


# Drop UAS columns
df = df.drop(columns =features_UAS_all, axis=1)
print('df length after UAS 

####################################################
# Splitting the data in train and test data
## I made sure that the test data needs to be a recent data

# Sub-df containing entries from years 2010 (to early 2022)
df_recent = df[df['Year'] >= 2010]

my_test_size = int(np.round(0.1 * len(df), 0))
print("We have a total of", len(df), 'entries.')
print("We allocate 10% of the total number of entries to the test set; this amounts to:", my_test_size, 'entries.')

df_recent_train, df_test = train_test_split(df_recent ,test_size=my_test_size,random_state=1234)
print("We have ", len(df_test), 'entries in our test set')

df_train=pd.concat ([df_recent_train, df[df['Year'] < 2010]]) 
print("We have ", len(df_train), 'entries in our train set')
df_train.shape


#################################################
# Saving the model 
##########################################################
# WARNING!! 
# If you execute this cell, you will OVERWRITTE the data!
##########################################################

%cd /content/drive/MyDrive/data/

# #to save the data
# # Save TEST data
with open("test_data_final.pkl", "wb") as f:
    pkl.dump([df_test], f) # saves the variables into a list

# # Save TRAIN data
with open("train_data_final.pkl", "wb") as f:
     pkl.dump([df_train], f) # saves the variables into a list
     