import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feature_columns=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

@st.cache()
def prediction(model,feature_columns):
  glass_type=model.predict([feature_columns])
  glass_type=glass_type[0]
  if glass_type == 1:
    return "building windows float processed".upper()

  elif glass_type == 2:
    return "building windows non float processed".upper()

  elif glass_type == 3:
    return "vehicle windows float processed".upper()

  elif glass_type == 4:
    return "vehicle windows non float processed".upper()

  elif glass_type == 5:
    return "containers".upper()

  elif glass_type == 6:
    return "tableware".upper()

  else:
    return "headlamp".upper()

    # S4.1: Add title on the main page and in the sidebar.
st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Analysis')

# S5.1: Using the 'if' statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox('show raw data'):
  st.subheader('Glass Type Data Set')
  st.dataframe(glass_df)

  # S6.1: Scatter Plot between the features and the target variable.
# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader('Scatter Plot')
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
feature_list=st.sidebar.multiselect('select the x axis',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# S6.2: Create scatter plots between the features and the target variable.
# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)
for feature in feature_list:
  st.subheader(f'Scatter Plot Between {feature} and Glass Type')
  plt.figure(figsize=(10,6))
  sns.scatterplot(x=feature,y='GlassType',data=glass_df)
  st.pyplot()

  # Sidebar for histograms.
st.sidebar.subheader('Histogram')
# Choosing features for histograms.
feature_hist=st.sidebar.multiselect('select the feature variable for histogram',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create histograms.
for feature in feature_hist:
  st.subheader(f'Histogram for {feature}')
  plt.figure(figsize=(10,6))
  plt.hist(glass_df[feature],bins='sturges',edgecolor='black')
  st.pyplot()

  #Create box plots for all the columns.
# Sidebar for box plots.
st.sidebar.subheader('Box Plot')
# Choosing columns for box plots.
feature_box=st.sidebar.multiselect('select the feature variable for box plot',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create box plots.
for feature in feature_box:
  st.subheader(f'Box Plot for {feature}')
  plt.figure(figsize=(10,6))
  sns.boxplot(glass_df[feature])
  st.pyplot()