import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Drinking_water1.csv")
#data2 = pd.read_csv("Drinking_water2.csv")

# filling the null values
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

#data2['ph'] = data2['ph'].fillna(data2['ph'].mean())
#data2['Sulfate'] = data2['Sulfate'].fillna(data2['Sulfate'].mean())
#data2['Trihalomethanes'] = data2['Trihalomethanes'].fillna(data2['Trihalomethanes'].mean())

# Select the features and target variable
X = data.drop(['id', 'Potability', 'Carcinogenics', 'medical_waste'], axis=1)
y = data['Potability']

# Select the features and target variable
#X2 = data2.drop(['id', 'Potability', 'Carcinogenics', 'medical_waste'], axis=1)
#y2 = data2['Potability']

# Train the ML models on the data
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

xgb_model = XGBClassifier()
xgb_model.fit(X, y)
#xgb_model.fit(X2,y2)

knn_model = KNeighborsClassifier()
knn_model.fit(X, y)

lr_model = LogisticRegression()
lr_model.fit(X, y)

svm_model = SVC()
svm_model.fit(X, y)

# Define the Streamlit app
def app():
    st.title("Water Potability Prediction")
    st.write("Choose a machine learning algorithm to predict the potability of drinking water based on various features.")
    algorithm = st.selectbox("Select algorithm", ["Random Forest", "XGBoost", "KNN", "Logistic Regression", "SVM"])

    # Add input widgets for the user to enter feature values
    ph = st.number_input("pH value", value=0.0)
    hardness = st.number_input("Hardness", value=0.0)
    solids = st.number_input("Solids", value=0.0)
    chloramines = st.number_input("Chloramines", value=0.0)
    sulfate = st.number_input("Sulfate", value=0.0)
    conductivity = st.number_input("Conductivity", value=0.0)
    organic_carbon = st.number_input("Organic Carbon", value=0.0)
    trihalomethanes = st.number_input("Trihalomethanes", value=0.0)
    turbidity = st.number_input("Turbidity", value=0.0)

    # Make a prediction using the chosen model
    if algorithm == "Random Forest":
        prediction = rf_model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    elif algorithm == "XGBoost":
        prediction = xgb_model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    elif algorithm == "KNN":
        prediction = knn_model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    elif algorithm == "Logistic Regression":
        prediction = lr_model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    elif algorithm == "SVM":
        prediction = svm_model.predict([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

    # Display the prediction to the user
    if prediction[0] == 0:
        st.write("The water is not potable.")
    else:
        st.write("The water is potable.")

    # Calculate the accuracy of the chosen model on the entire dataset
    if algorithm == "Random Forest":
        y_pred = rf_model.predict(X)
    elif algorithm == "XGBoost":
        y_pred = xgb_model.predict(X)
    elif algorithm == "KNN":
        y_pred = knn_model.predict(X)
    elif algorithm == "Logistic Regression":
        y_pred = lr_model.predict(X)
    elif algorithm == "SVM":
        y_pred = svm_model.predict(X)


    #accuracy = accuracy_score(y, y_pred)
    #st.write("The accuracy of the model on the training data is:", accuracy)

# Run the Streamlit app
if __name__ == '__main__':
    app()

