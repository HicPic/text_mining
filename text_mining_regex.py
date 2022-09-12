#######################
# Import packages
#######################
import numpy as np
import seaborn as sns

#######################
# Pandas
#######################
import pandas as pd
# Set pandas settings to show all data when using .head(), .columns etc.
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.colheader_justify","left") # left-justify the print output of pandas

### Display full columnwidth
# Set pandas settings to display full text columns
#pd.options.display.max_colwidth = None
# Restore pandas settings to display standard colwidth
pd.reset_option('display.max_colwidth')

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

#####################
# NLP 
#####################
import re # for Regular Expression handling
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # WordNet lemmatizer
nltk.download('omw-1.4') # necessary for WordNet lemmatizer
from nltk.tokenize import word_tokenize # Usual tokenizer
from nltk.tokenize import TweetTokenizer # Special tokenizer;  "we'll", "didn't", etc. are considered as one word
from sklearn.feature_extraction.text import CountVectorizer # Vectorization
from nltk.corpus import stopwords # Import stopwords from nltk.corpus


###############################
# Load the data
###############################
#@title
# Load the TRAIN data (97417 entries)
# Do not touch the TEST data until the end of the project!
# or the curse of the greek gods will fall upon you!

%cd /content/drive/MyDrive/data/transformed/
with open("train_data_final.pkl", "rb") as f:
    loaded_data = pkl.load(f)

df = loaded_data[0]
print("\nA Dataframe with", len(df), "entries has been loaded")


################################
# Compile regular expressions
###############################

# Compile Regular Expressions
r_R_L_C = re.compile(r"""
                  (?i)                   # turns on the case-insensitive mode of RegEx
                  \d+R 
                  | 
                  \d+L                   # at least a number *before* 'R' or 'L' or 'C'
                  |
                  \d+C
                  """, re.VERBOSE)

r_Right_Left_Center = re.compile(r"""
                  (?i)                   # turns on the case-insensitive mode of RegEx
                  \d+ Right 
                  | 
                  \d+ Left               # at least a number *before*
                  |
                  \d+ Center
                  """, re.VERBOSE)

r_Celcius_neg = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \s                            # Avoid finding an aircraft model type, e.g. 'MH-65C'
                  -                             # Our hypothesis: most temp. indications are negative
                                                # otherwise confustion with RWY indication '36C' (center)
                  \d+                           
                  C
                  \s                            
                """, re.VERBOSE)

r_Celcius_degrees = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d+
                  \s                           
                  degrees                       # appears mostly in plural form
                  \s* 
                  C
                  \s                            # without this, you will match '120 degree clearing turn' 
                """, re.VERBOSE)

r_Mach_1 = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  Mach
                  \s*
                  0*\.\d{1,}                    # decimal number *after* 'Mach'
                  """, re.VERBOSE)

r_Mach_2 = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  0*\.\d{1,}                    # decimal number *before* 'Mach'
                  \s*
                  Mach
                  """, re.VERBOSE)

r_gust = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d+
                  g
                  \d+
                  """, re.VERBOSE)

# 0000Z format
r_Z_1 = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \s
                  \d{4,4}                       # exactly 4 digits preceeding Zulu time (UTC +0) 'Z'    
                  Z
                  \s                            
                  """, re.VERBOSE)

# 00:00Z format
r_Z_2 = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \s
                  \d{2,2}                       # exactly 2 digits     
                  \:
                  \d{2,2}
                  Z
                  \s                            
                  """, re.VERBOSE)

r_KTS = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d+
                  KTS
                  |
                  \d+
                  KT                        
                """, re.VERBOSE)

r_alt_1 = re.compile(r"""
                    (?i)                         # turns on the case-insensitive mode of RegEx
                    FL\d{1,}     
                    """, re.VERBOSE)


r_alt_2 = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d{0,2}[\.\,\;]
                  \d*
                  FT
                  """, re.VERBOSE)

r_am_num = re.compile(r"""                      # RegEx for 'am' or 'AM' preceeded by numbers
                  (?i)                          # turns on the case-insensitive mode of RegEx                 
                  [0-1X\w]{1,2}
                  [\;\:\,\.]*
                  [0-59]{1,2}
                  am                
                  """, re.VERBOSE)

r_pm = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx                 
                  [0-1X\w]{1,2}
                  [\;\:\,\.]*
                  [0-59]{1,2}
                  pm                
                  """, re.VERBOSE)

r_dist_MI = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d                            # at least one digit, sticked to 'mi'
                  MI
                  """, re.VERBOSE)

r_dist_SM = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d                            # at least one digit, sticked to 'mi'
                  SM
                  """, re.VERBOSE)

r_dist_NM = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d                            # at least one digit, sticked to 'mi'
                  NM
                  """, re.VERBOSE)

r_MIN = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d+                           # at least one digit. Without '+', it maches only '0minute' from '20minutes'
                  MINUTES                        
                  |
                  \d+
                  MINUTE
                  |
                  \d+
                  MINS
                  | 
                  \d+
                  MIN
                  """, re.VERBOSE)

r_HR = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d+
                  HR                        
                  """, re.VERBOSE)

r_LBS = re.compile(r"""
                  (?i)                          # turns on the case-insensitive mode of RegEx
                  \d+
                  LBS                        
                  """, re.VERBOSE)

# Define replacements to perform
# All the findings of the RegEx's in the list will be substituted by the corresponding key
# Note the precise use a spaces the substitute, to avoid problems during tokenization
my_dict = {
          ' <RUNWAY> ': [r_R_L_C, r_Right_Left_Center],
          ' <TEMP_C_NEG> ': [r_Celcius_neg],
          ' <TEMP_C_POS> ': [r_Celcius_degrees],
          ' Mach ': [r_Mach_1, r_Mach_2],
          ' <GUST> ': [r_gust],
          ' <TIME> ': [r_Z_1, r_Z_2],
          ' KT ': [r_KTS],
          ' FT ': [r_alt_1, r_alt_2],
          ' Ante_Meridiem ': [r_am_num],
          ' PM ': [r_pm],
          ' MI': [r_dist_MI],
          ' SM ': [r_dist_SM],
          ' NM ': [r_dist_NM],
          ' MIN': [r_MIN],           # NO space at the end of ' MIN'
          ' HR': [r_HR],             # NO space at the end of ' HR'
          ' LBS ': [r_LBS]
          }


def substitute_RegEx(my_dict):
  """
  Inputs: a dictionnary with RegEx's and subsitutes
  Go through the narratives and replace the findings of the RegEx's by the substitutes 
  passed as input (keys of the dictionnary).
  Write the new version of the narrative after the replacements into a new column of df 
  entitled 'Narrative_RegEx_subst'.
  """
  
  # Time function execution
  import time
  start_time = time.time()
  print(7*'-', "Execution started, why don't you grab a coffee...", 7*'-', '\n')

  # Copy the narratives into a new column
  df['Narrative_RegEx_subst'] = df['Narrative']

  # Initialize counters
  repl_counter = 0
  progress = 0

  # Loop through the narratives using their index (here ACN number)
  for idx in df['Narrative_RegEx_subst'].index:
    
    # Loop through the keys of the dict
    for k in my_dict.keys():  
      
      # Loop through the list of RegEx's that correspond to that key
      for regex in my_dict[k]:
        new_term = k
        repl_result = re.subn(regex, new_term, df['Narrative_RegEx_subst'].loc[idx])
        # The re.subn() method returns the new version of the target string after the replacements 
        # The second element is the number of replacements it has made
    
        # New version of the narrative, after the replacements
        df['Narrative_RegEx_subst'][idx] = repl_result[0]

        # Increment the counter of remplacements by the number of replacements done in the narrative
        repl_counter = repl_counter + repl_result[1]
      
    # Report on the execution progress  
    progress += 1
    if progress % 1000 == 0:
      print(f"{progress} narratives processed; {repl_counter} replacements done so far \n")

  end_time = time.time()
  print(f"--- Executed in {np.round((end_time - start_time)/60,1)} minutes ---")
  print(f"{repl_counter} replacements in total")

  return None
  
  # Call the function
# /!\ takes about 50min. to execute!
substitute_RegEx(my_dict)


##########################################################
# Save the output externally
# WARNING!! 
# If you execute this cell, you will OVERWRITTE the data!
##########################################################

%cd /content/drive/MyDrive/data/

# save the df['Narrative_RegEx_subst'] externally to avoid having to perform the 
# substitutions again

with open("Narrative_RegEx_subst_21072022_TRAIN.pkl", "wb") as f:
    pkl.dump([df['Narrative_RegEx_subst']], f) # saves the variables into a list  
