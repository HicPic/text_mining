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

# Package to show the progression of pandas operations
from tqdm import tqdm
# from tqdm.auto import tqdm  # for notebooks

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()
# simply use .progress_apply() instead of .apply() on your pd.DataFram

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
from nltk.stem.snowball import EnglishStemmer

######################
# Naive bayes
######################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

##############################
# Undersampeling
##############################
from imblearn.under_sampling import RandomUnderSampler 

###############################
# ML preprocessing and models
###############################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble # random forest
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

import pickle as pkl # Saving data externally


##########################
# LOAD THE DATA
##########################
#@title
# Load the TRAIN data (97417 entries)
# Do not touch the TEST data until the end of the project!
# or the curse of the greek gods will fall upon you!

%cd /content/drive/MyDrive/data/
with open("train_data_final.pkl", "rb") as f:
    loaded_data = pkl.load(f)

df = loaded_data[0]
print("\nA Dataframe with", len(df), "entries has been loaded")


# Load the file Narrative_RAW_stemmed_21072022_TRAIN.pkl 
%cd /content/drive/MyDrive/data/transformed/
with open("Narrative_Raw_Stemmed_24072022_TRAIN.pkl", "rb") as f:
    loaded_narr = pkl.load(f)

df['Narrative_Raw_Stemmed'] = loaded_narr[0] 
print("Data loaded")

# Load the file Narrative_PP_stemmed_21072022_TRAIN.pkl
%cd /content/drive/MyDrive/data/transformed/
with open("Narrative_PP_stemmed_24072022_TRAIN.pkl", "rb") as f:
    loaded_narr = pkl.load(f)

df['Narrative_PP_stemmed'] = loaded_narr[0]
print("Data loaded")


###########################################@
# Create sparse matrix from row and PP narratives using count vectorizer

# RAW NARRATIVES
# Join all tokens into a single string
df['Narrative_Raw_Stemmed_str'] = df['Narrative_Raw_Stemmed'].apply(lambda token_list: ' '.join(entry for entry in token_list))

# Instantiate the vectorizer with default settings
vectorizer = CountVectorizer() 

# Sparse matrix
spm = vectorizer.fit_transform(df['Narrative_Raw_Stemmed_str']) # scipy.sparse.csr_matrix object

# Instantiate a second CountVectorizer with the the binary = True option 
vectorizer_bool = CountVectorizer(binary = True) # all non zero counts are set to 1

# Sparse matrix in boolean form
spm_bool = vectorizer_bool.fit_transform(df['Narrative_Raw_Stemmed_str'])

# Vocabulary
vocab = vectorizer.get_feature_names_out()
# ordered by the word indices
print(f"Vocabulary length: {len(vocab)}")
   
# PP NARRATIVES
# Join all tokens into a single string
df['Narrative_PP_Stemmed_str'] = df['Narrative_PP_stemmed'].apply(lambda token_list: ' '.join(entry for entry in token_list))

# Instantiate the vectorizer with default settings
vectorizer = CountVectorizer() 

# Sparse matrix
spm = vectorizer.fit_transform(df['Narrative_PP_Stemmed_str']) # scipy.sparse.csr_matrix object

# Instantiate a second CountVectorizer with the the binary = True option 
vectorizer_bool = CountVectorizer(binary = True) # all non zero counts are set to 1

# Sparse matrix in boolean form
spm_bool = vectorizer_bool.fit_transform(df['Narrative_PP_Stemmed_str'])

# Vocabulary
vocab = vectorizer.get_feature_names_out()
# ordered by the word indices
print(f"Vocabulary length: {len(vocab)}")


##############################################
# Target Feature engineering
############################################
# Drop the NaNs in Anomaly
print(f"We have {len(df[df['Anomaly'].isna()])} entries where 'Anomaly' == NaN. We drop these entries")

df = df.dropna(axis = 0, how = 'any', subset = ['Anomaly'])
print(f"Current length of our DataFrame: {len(df)}")

# One-hot encode Anomaly root labels
# Root label (source = ASRS coding forms)
Anomaly_RootLabels=['Aircraft Equipment',
                    'Airspace Violation',
                    'ATC Issue',
                    'Flight Deck / Cabin / Aircraft Event',
                    'Conflict',
                    'Deviation - Altitude',
                    'Deviation - Speed',
                    'Deviation - Track / Heading',
                    'Deviation / Discrepancy - Procedural',
                    'Ground Excursion',
                    'Ground Incursion',
                    'Ground Event / Encounter',
                    'Inflight Event / Encounter',
                    'No Specific Anomaly Occurred']

# Create a column in the df corresponding to each Anomaly root label 'anomaly_rl'
Anomaly_RootLabels_columns = []
for anomaly_rl in Anomaly_RootLabels:
    col='Anomaly_' + anomaly_rl
    Anomaly_RootLabels_columns.append(col)
    # Fill the columns in a one-hot-encoding logic
    df[col] = df['Anomaly'].astype('str').apply(lambda x: 1 if (anomaly_rl in x)  else 0)

#########################################
# Naive Bayes Model
#########################################

def Naive_bayes(data, target):
  """
  Return the predictions metrics for Naive bayes

  Inputs:
  - data: pd.Series containing the narratives as single string
  - target: ndarray of a shape(#samples, #classes) containing 0's and 1's
            A sample may contain several 1's (multilabel context)
            min_df: the value of min_df used

  Returns: 
  - y_pred: The multilabel predictions of our model, i.e. ndarray
    of shape (#samples, #classes) containing probabilities to belong to each class,
    For a given sample, the sum of probabilities may exceed 1 (multilabel context)

  - classification report

  """
  #print ('the current min df is:', min_df)
  # Train-test split  
  X_train, X_test, y_train, y_test = train_test_split(data, target, 
                                                      test_size= 0.2, 
                                                      random_state = 12)
  
  #define a list of min_dfs
  min_dfs = [12, 50, 100,150, 200, 600, 1000, 1600]
  d = pd.DataFrame()
  NB_pipeline_opts = pd.DataFrame()
  for min_df in min_dfs:
    # Define a pipeline combining a text feature extractor with multi lable classifier
    NB_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
                                min_df = min_df)),
                    ('clf', OneVsRestClassifier(MultinomialNB(
                        fit_prior=True, class_prior=None))),
                ])
  
    # Train the model using X_train and y_train_Anomaly
    NB_pipeline.fit(X_train,y_train)

    # Compute the testing accuracy 
    y_pred = NB_pipeline.predict(X_test)
    
    # classification report 
    data = classification_report(y_test, y_pred, 
                                  target_names = Anomaly_RootLabels_columns, output_dict=True)
    report = pd.DataFrame(data)
    NB_pipeline_opts = pd.concat([NB_pipeline_opts ,pd.Series(np.repeat(min_df,4))])
    d = pd.concat([d,report])   
    
  NB_pipeline_opts = NB_pipeline_opts.reset_index()
  NB_pipeline_opts = NB_pipeline_opts.rename(columns ={0:'min_df'})
  d = d.reset_index()
  d = d.rename(columns ={'index':'metric'})

  df_reports = pd.concat([NB_pipeline_opts, d], axis=1)
  df_reports = df_reports.rename({0:'min_df'}, axis=1)
  return df_reports
  
  
  # initialize the objects necesseray function inputs
data = df['Narrative_PP_Stemmed_str']
target = df[Anomaly_RootLabels_columns].values
#for min_df in min_dfs:
naive_bayes = Naive_bayes(data, target)


#Append the string 'min_df' to the Anomaly_root_label_columns to use it for indexing
#Anomaly_RootLabels_columns = Anomaly_RootLabels_columns.append('min_df')

#Indexing a new table with only the f-scores
naive_fscore = naive_bayes[(naive_bayes['metric'] =='f1-score')][Anomaly_RootLabels_columns]
naive_fscore.columns


# PLOTTING 
# Line plots of all anomalies into one chart
plt.figure(figsize =(20,15))
naive_fscore.plot(x= "min_df",y= Anomaly_RootLabels_columns, kind="line", figsize=(30,10))
# set legend position
plt.legend(bbox_to_anchor=(1, 1))

plt.title('Naive bayes: Anomalies f1-score according to the vovabulary length')
plt.ylabel('f1-score')
plt.xlabel('min_df value');


# Subplot (I used bar plot) for each anomaly
min_dfs = [12, 50, 100,150, 200, 600, 1000, 1600]
plt.figure(figsize = (20,70))
for i,anomaly in enumerate(Anomaly_RootLabels_columns):
  i =i+1
  plt.subplot(7,2,i)
  plt.bar(range(len(min_dfs)), 
        naive_fscore[anomaly], 
        label = anomaly)
  plt.xticks(range(len(min_dfs)), min_dfs)
  plt.ylabel('f1_score')
  plt.xlabel('min_dfs')
  plt.title(anomaly);  
