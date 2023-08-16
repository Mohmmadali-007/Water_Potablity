import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Drinking_water.csv')
    return data

data = load_data()

# filling the null values
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Select the features and target variable
x = data.drop(['Potability'], axis=1)
y = data['Potability']

print(x.shape)
print(y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a Streamlit app
st.title('Water Potability Prediction App')
st.write('Enter the following parameters to predict water portability')

ph = st.number_input('pH value')
hardness = st.number_input('Hardness')
solids = st.number_input('Total Solids')
chloramines = st.number_input('Chloramines')
sulfate = st.number_input('Sulfate')
conductivity = st.number_input('Conductivity')
organic_carbon = st.number_input('Organic Carbon')
trihalomethanes = st.number_input('Trihalomethanes')
turbidity = st.number_input('Turbidity')

# Make a prediction using the model
prediction = model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

if prediction == 0:
    st.write('The water is not portable.')
else:
    st.write('The water is portable.')

# Display the model's performance
st.write('Accuracy:', accuracy)
st.write('Confusion Matrix:', conf_matrix)
