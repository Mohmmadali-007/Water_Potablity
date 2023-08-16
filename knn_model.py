import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Drinking_water.csv")

# filling the null values
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Select the features and target variable
X = data.drop(['Potability'], axis=1)
y = data['Potability']

# Train a KNN classifier on the data
model = KNeighborsClassifier()
model.fit(X, y)

# Define the Streamlit app
def app():
    st.title("My Machine Learning App")
    st.write("This app uses a K-Nearest Neighbors classifier to predict the target variable based on the input features.")

    ph = st.number_input("pH value", value=0.0)
    Hardness = st.number_input("Hardness", value=0.0)
    Solids = st.number_input("Solids", value=0.0)
    Chloramines = st.number_input("Chloramines", value=0.0)
    Sulfate = st.number_input("Sulfate", value=0.0)
    Conductivity = st.number_input("Conductivity", value=0.0)
    Organic_carbon = st.number_input("Organic Carbon", value=0.0)
    Trihalomethanes = st.number_input("Trihalomethanes", value=0.0)
    Turbidity = st.number_input("Turbidity", value=0.0)

    # Make a prediction using the trained model
    prediction = model.predict([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

    # Display the prediction to the user
    if prediction[0] == 0:
        st.write("The water is not potable.")
    else:
        st.write("The water is potable.")

     # Calculate the accuracy of the model on the entire dataset
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write("The accuracy of the model on the training data is:", accuracy)

# Run the Streamlit app
if __name__ == '__main__':
    app()
