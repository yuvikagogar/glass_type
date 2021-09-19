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

# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader('Visualisation Selector')
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_type=st.sidebar.multiselect('Select the Charts/Plots:',('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
st.set_option('deprecation.showPyplotGlobalUse', False)
  # Sidebar for histograms.
if 'Histogram' in plot_type:
  st.subheader('Histogram')
  columns=st.sidebar.selectbox('select the column to create histogram',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(10,6))
  plt.title(f'histogram for {columns}')
  plt.hist(glass_df[columns],bins='sturges',edgecolor='black')
  st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
  #Create box plots for all the columns.
# Sidebar for box plots.
if 'Box Plot' in plot_type:
  st.subheader('Box Plot')
  columns=st.sidebar.selectbox('select the column to create box plot',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(10,6))
  plt.title(f'boxplot for {columns}')
  sns.boxplot(glass_df[columns])
  st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
  #Create count plot, pie chart, correlation heatmap and pair plot.
# Create count plot using the 'seaborn' module and the 'st.pyplot()' function.
if 'Count Plot' in plot_type:
  st.subheader('Count Plot')
  sns.countplot(x='GlassType',data=glass_df)
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.   
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Pie Chart' in plot_type:
  st.subheader('Pie Chart')
  pie_data=glass_df['GlassType'].value_counts()
  plt.figure(figsize=(5,5))
  plt.pie(pie_data,labels=pie_data.index,autopct='%1.2f%%',startangle=30,explode=np.linspace(.06,.16,6))
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Correlation Heatmap' in plot_type:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize=(10,5))
  ax=sns.heatmap(glass_df.corr(),annot=True)
  bottom,top=ax.get_ylim()
  ax.set_ylim(bottom+0.5,top-0.5)
  st.pyplot()
# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function. 
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Pair Plot' in plot_type:
  st.subheader('Pair Plot')
  sns.pairplot(glass_df)
  st.pyplot()

st.sidebar.subheader('select your values')
ri=st.sidebar.slider(' Input Ri',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na=st.sidebar.slider(' Input Na',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg=st.sidebar.slider(' Input Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al=st.sidebar.slider(' Input Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
si=st.sidebar.slider(' Input Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
k=st.sidebar.slider(' Input K',float(glass_df['K'].min()),float(glass_df['K'].max()))
ca=st.sidebar.slider(' Input Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
ba=st.sidebar.slider(' Input Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
fe=st.sidebar.slider(' Input Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))

# Add a subheader in the sidebar with label "Choose Classifier"
st.sidebar.subheader('choose classifier')
# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Random Forest Classifier','LogisticalRegression'))

cll=st.sidebar.button('classifier')
if classifier == 'Support Vector Machine':
  st.sidebar.subheader('model hyperparameter')
  c_value=st.sidebar.number_input('C(error rate)',1,100,step=1)
  kernel_input=st.sidebar.radio('kernel',('linear','rbf','poly'))
  gamma_input=st.sidebar.number_input('gamma',1,100,step=1)
    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
  if cll:
    st.subheader('support vector machine')
    svc_model=SVC(C=c_value,kernel=kernel_input,gamma=gamma_input)
    svc_model.fit(X_train,y_train)
    y_predict=svc_model.predict(X_test)
    accuracy=svc_model.score(X_test,y_test)
    glass_type=prediction(svc_model,[ri,na,mg,al,si,k,ca,ba,fe])
    st.write('the type of the glass predicted is',glass_type)
    st.write('accuracy',accuracy.round(2))
    plot_confusion_matrix(svc_model,X_test,y_test)
    st.pyplot()

    if classifier == 'Random Forest Classifier':
      st.sidebar.subheader('Random Forest Classifier')
      n_es=st.sidebar.number_input('no of trees in forest',100,5000,step=10)
      max_dep=st.sidebar.number_input('max depth of tree',1,100,step=1)
      if cll:
        st.subheader('Random Forest Classifier')
        rf_clf=RandomForestClassifier(n_estimators=n_es,max_depth=max_dep,n_jobs=-1)
        rf_clf.fit(X_train,y_train)
        y_predict=rf_clf.predict(X_test)
        accuracy=rf_clf.score(X_test,y_test)
        glass_type=prediction(rf_clf,[ri,na,mg,al,si,k,ca,ba,fe])
        st.write('the type of the glass predicted is',glass_type)
        st.write('accuracy',accuracy.round(2))
        plot_confusion_matrix(rf_clf,X_test,y_test)
        st.pyplot()
        # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
        # This 'if' statement must be inside the above 'if' statement.
      #if st.sidebar.button('Classifier'):
        