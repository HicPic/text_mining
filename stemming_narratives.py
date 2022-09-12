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
# Define global plot parameters for better readability and consistency among plots
# A complete list of the rcParams keys can be retrieved via plt.rcParams.keys() function
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 23
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 23
plt.rc('legend', fontsize=23)    # legend fontsize

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
# ML preprocessing and models
###############################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix

import pickle as pkl # Saving data externally




###################################
#LOAD DATA FROM .PKL FILE
##################################

# Load the TRAIN data (97417 entries)
# Do not touch the TEST data until the end of the project!
# or the curse of the greek gods will fall upon you!

%cd /content/drive/MyDrive/data/transformed/
with open("train_data_final.pkl", "rb") as f:
    loaded_data = pkl.load(f)

df = loaded_data[0]
print("\nA Dataframe with", len(df), "entries has been loaded")


# Load the file Narrative_RegEx_subst_20072022_TRAIN.pkl
%cd /content/drive/MyDrive/data/transformed/
with open("Narrative_RegEx_subst_21072022_TRAIN.pkl", "rb") as f:
    loaded_narr = pkl.load(f)

df['Narrative_RegEx_subst'] = loaded_narr[0]
print("Data loaded")


# Import of known abbreviations data
df_abbrev = pd.read_csv('Qualified abbreviations_20220718.csv', low_memory=False,index_col=0, sep=';')
df_abbrev = df_abbrev[['Abbreviation','Full-Text','Forms to parse','Category2']]


# Deal with various forms of Abbreviations: 
## i.e. HR but also HRLY, HRS
## Initially, we have 1 column with then Abbreviation prefix ('HR') and 1 column with the different sub-forms (;S;LY) if they exist
## We fill up the column 'Abbreviation forms list' with the forms HR;HRS;HRLY
## This needs to be done in 2 steps:
### 1 First concatenates in 1 column abbreviations + forms : in 1 list [HR,S,LY]
### 2 Applies to this list a function that generates the forms : [HR,HRS,HRLY]


#Step1 : Concatenates in 1 column the abbreviation "HR" and its different forms ";S;LY" into a list [HR,S,LY]
df_abbrev['Abbreviation+Forms to parse']=df_abbrev['Abbreviation']+df_abbrev['Forms to parse'].astype('str').apply(lambda x: '' if x=='nan' else x)

def get_abbreviation_forms(abb_and_forms):
  """ 
  Parameters = 1 list of with Abbreviation , followed, if they exist, by its different forms' suffix : HR;S;LY
  Returns : a list of the abbreviation forms [HR,HRS,HRLY]
  """
  full_abbrev_forms=[]
  parsed_abbrev_forms=abb_and_forms.split(';')
  abbrev=parsed_abbrev_forms[0]
  full_abbrev_forms.append(abbrev)
  if len (parsed_abbrev_forms)>1:
    for form in parsed_abbrev_forms[1:]:
      abbrev_form=abbrev+form
      full_abbrev_forms.append(abbrev_form)
  return full_abbrev_forms
  
  #Step2 : Uses the list  [HR,S,LY]  to compose the forms [HR,HRS,HRLY]
df_abbrev['Abbreviation forms list']=df_abbrev['Abbreviation+Forms to parse'].apply(get_abbreviation_forms)


# Example for abbreviations with different forms
df_abbrev[df_abbrev['Forms to parse'].isna()==False].head()

# Example for abbreviations with only 1 form 
df_abbrev[df_abbrev['Forms to parse'].isna()].head()


# On narratives : Expressions to Abbreviations (Air traffic control => ATC)
##All expressions except for Intermediate Fix IF and Second Officer SO

# We focus on expressions except for IF & SO
df_abbrev_expressions=df_abbrev[(df_abbrev['Category2']=='Expression')&(df_abbrev['Abbreviation']!='IF')&(df_abbrev['Abbreviation']!='SO')]
len(df_abbrev_expressions)

df['Narrative_Expression_subst']=df['Narrative_RegEx_subst']
for row in df_abbrev_expressions.index :
  # List of Full-Text forms for each abbreviation
   parsed_full_text=df_abbrev_expressions['Full-Text'][row].split(';') # Expression to replace
   abbreviation=df_abbrev_expressions['Abbreviation'][row] # Abbreviation as replacement
   for ft in parsed_full_text :
     df['Narrative_Expression_subst']=df['Narrative_Expression_subst'].astype('str').apply(lambda x: x.lower().replace(ft.lower(),abbreviation.lower()))

# We focus on expressions Intermediate Fix (IF) & Second Officer (SO) : we add '<' '>' to the abbreviation (<IF>,<SO>) to avoid confusion with the words if and so
df_abbrev_expressions=df_abbrev[(df_abbrev['Abbreviation']=='IF')|(df_abbrev['Abbreviation']=='SO')]
len(df_abbrev_expressions)

for row in df_abbrev_expressions.index :
  # List of Full-Text forms for each abbreviation
   parsed_full_text=df_abbrev_expressions['Full-Text'][row].split(';') # Expression to replace
   abbreviation=' <'+ df_abbrev_expressions['Abbreviation'][row]+ '> ' # Abbreviation as replacement
   for ft in parsed_full_text :
     df['Narrative_Expression_subst']=df['Narrative_Expression_subst'].astype('str').apply(lambda x: x.lower().replace(ft.lower(),abbreviation.lower()))
     
     
# Further step: find new N-grams and qualify them as Expressions
## We should detect "Mach trim" "Yaw damper" "Feet MSL" previously identified during our researches

# Lower case tokenization
# Create a new column in the df with the narratives in lower case and tokenized (Tweet Tokenizer)
# Purpose: avoid to tokenize each narrative for each abbreviation : 500 * 100 000  
# Takes ~ 3 min. to execute
Tweet_tokenizer=TweetTokenizer()
starting_time = time() 
df['Narrative_tokenized']=df['Narrative_Expression_subst'].apply(lambda x:Tweet_tokenizer.tokenize(x.lower()))

ending_time = time()
exec_time = ending_time - starting_time
print("Calculation took", exec_time/60, "mns to run")

# We focus on contractions except for AM (Ante Meridiem) 
df_abbrev_contractions=df_abbrev[(df_abbrev['Category2']=='Contraction')&(df_abbrev['Abbreviation']!='AM')]
len(df_abbrev_contractions)

df['Narrative_tokenized_Contraction_subst']=df['Narrative_tokenized']

# We loop on all contractions
for row in df_abbrev_contractions.index :
  # List of all contraction forms for each abbreviation ie HR,HRS,HRLY
   contractions_list=df_abbrev_contractions['Abbreviation forms list'][row] # List of Contractions to replace
   full_text=df_abbrev_contractions['Full-Text'][row].lower() # Full_text as replacement ie 'hour''
   for contraction in contractions_list :
     # we loop on the list on the list of contractions to replace EXACTLY (==): 
     # ex tokens  HR, HRS, HRLY  are replaced by 'hour'  
     # but 'HR12' would not be replaced by 'hour12'
      df['Narrative_tokenized_Contraction_subst']=df['Narrative_tokenized_Contraction_subst'].apply(lambda list_of_tokens: [full_text if token==contraction.lower() else token for token in list_of_tokens])

#############################################################
# STOP WORD FILTERING
######################
# Initialiser la variable des mots vides
stop_words = set(stopwords.words('english')) 
# we convert it to a set, more efficient (vs. a liste) 
# when searching for stopwords. A set also removes duplicates.
# add additional stop words, if desired : 
stop_words.update(['.', ';', '[', ']', '(', ')',"'","@",
                    "they've", "they're", "they'll", 
                    "i've", "i'm", "i'll", "could"])

def stop_words_filtering(my_list):
    """
    Delete stop words from the function input ‘my_list’.
    Keeps multiple occurrences of Non-stop words for the Bag of words approach
    """

    for stopword in stop_words:

        if stopword in my_list:
            # Uses filter function to remove all occurrences of stopwords
            my_list=list(filter(lambda word: word != stopword, my_list))
    return my_list

#############################################################
starting_time = time() 
df['Narrative_SW_filtered']=df['Narrative_tokenized_Contraction_subst'].apply(lambda tokens:stop_words_filtering(tokens))

ending_time = time()
exec_time = ending_time - starting_time
print("Calculation took", exec_time/60, "mns to run")


#############################################################
 # STEMMING
 ######################
def stemming(tokens):
    """
    Stem the list of tokens passed as input
    """
    stemmer = EnglishStemmer()
    radicals = []
    
    for word in tokens:
        radical = stemmer.stem(word)
        radicals.append(radical)

#    return set(radicals) # remove any duplicates
    return radicals 
    
    starting_time = time() 
df['Narrative_PP_stemmed']=df['Narrative_SW_filtered'].apply(lambda tokens:stemming(tokens))

ending_time = time()
exec_time = ending_time - starting_time
print("Calculation took", exec_time/60, "mns to run")

##########################################################
# Save the output externally
# WARNING!! 
# If you execute this cell, you will OVERWRITTE the data!
##########################################################

%cd /content/drive/MyDrive/data/transformed/

# save the df['Narrative_RegEx_subst'] externally to avoid having to perform the 
# substitutions again
with open("Narrative_PP_stemmed_24072022_TRAIN.pkl", "wb") as f:
    pkl.dump([df['Narrative_PP_stemmed']], f) # saves the variables into a list
    

#########################################################
# Direct Tokenization / Stop Word filtering / Stemming on raw narratives
########################################################
df['Narrative_Raw_Stemmed']=df['Narrative'].apply(lambda x:Tweet_tokenizer.tokenize(x.lower()))
df['Narrative_Raw_Stemmed']=df['Narrative_Raw_Stemmed'].apply(lambda tokens:stop_words_filtering(tokens))
df['Narrative_Raw_Stemmed']=df['Narrative_Raw_Stemmed'].apply(lambda tokens:stemming(tokens))


##########################################################
# Save the output externally
# WARNING!! 
# If you execute this cell, you will OVERWRITTE the data!
##########################################################
%cd /content/drive/MyDrive/data/transformed/

# save the df['Narrative_RegEx_subst'] externally to avoid having to perform the 
# substitutions again
with open("Narrative_Raw_Stemmed_24072022_TRAIN.pkl", "wb") as f:
    pkl.dump([df['Narrative_Raw_Stemmed']], f) # saves the variables into a list

