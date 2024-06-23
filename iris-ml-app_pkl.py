# Codes adapted from https://github.com/dataprofessor/code/tree/master/streamlit/part2

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Markdown string 
# st.write("markdown string")
# controls font size. None means Body, 1 means heading, more # means smaller heading 
# In this case, st.write() is used for the heading and subtitle for the website
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

# Title for sidebars  
st.sidebar.header('User Input Parameters')

# Function to gather inputs for prediction 
# Data is collected via sliders in the sidebar: .slider("title", min, max, default)
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    # Data captured in dictionary to be convered to DataFrame
    # Keys must match columns names in the dataset for saving
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    # Inputs saved as a DataFrame 
    features = pd.DataFrame(data, index=[0])
    # Function returns user's inputs
    return features

# Store user's input from the function into df variable 
df = user_input_features()

# Subheader to display user's entered values 
st.subheader('User Input parameters')
# Display the resultant DataFrame 
st.write(df)

# Load dataset
iris = datasets.load_iris()

# *By right, you should use train_test_split()

# Separate features and target
X = iris.data
y = iris.target

# Define model
clf = RandomForestClassifier()
# Train model
clf.fit(X,y)

# Apply model to make predictions
prediction = clf.predict(df)
# Returns prediction probability. Probability (percentage decimal) of each class being the result 
# The model's confidence. Eg: It is _% sure it is _
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)