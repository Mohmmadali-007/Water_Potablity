import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("Drinking_water.csv")

# filling the null values
data['ph'] = data['ph'].fillna(data['ph'].mean())
data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

# Select the features and target variable
X = data.drop(['id', 'Potability', 'Carcinogenics', 'medical_waste'], axis=1)
y = data['Potability']

# Perform data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM model
model = SVC()

# Train the model
model.fit(X_train, y_train)

# Define the Streamlit app
def app():
    st.title('Water Potability Prediction')
    
    # Get the user input
    ph = st.slider('pH Value', 0.0, 14.0, 7.0)
    hardness = st.number_input('Hardness Value')
    solids = st.number_input('Total Solids')
    chloride = st.number_input('Chloride Content')
    sulfate = st.number_input('Sulfate Content')
    conductivity = st.number_input('Conductivity')
    organic_carbon = st.number_input('Organic Carbon Content')
    trihalomethanes = st.number_input('Trihalomethanes Content')
    turbidity = st.number_input('Turbidity')
    
    # Make the prediction
    input_data = [[ph, hardness, solids, chloride, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]]
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    # Display the prediction
    if prediction[0] == 0:
        st.write('The water is not potable')
    else:
        st.write('The water is potable')

if __name__ == '__main__':
    app()
