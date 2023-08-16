import streamlit as st
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


st.write("""
# Water Potability App
This app predicts if water is potable based on 9 variables.
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    pH = st.sidebar.number_input('pH', 0.22, 14.0, 7.0)
    Hardness = st.sidebar.number_input('Hardness')
    Solids = st.sidebar.number_input('Solids')
    Chloramines = st.sidebar.number_input('Chloramines')
    Sulfate = st.sidebar.number_input('Sulfate')
    Conductivity = st.sidebar.number_input('Conductivity')
    Organic_carbon = st.sidebar.number_input('Organic carbon')
    Trihalomethanes = st.sidebar.number_input('Trihalomethanes')
    Turbidity = st.sidebar.number_input('Turbidity')

    data = {
        'pH': pH,
        'Hardness': Hardness,
        'Solids': Solids,
        'Chloramines': Chloramines,
        'Sulfate': Sulfate,
        'Conductivity': Conductivity,
        'Organic_carbon': Organic_carbon,
        'Trihalomethanes': Trihalomethanes,
        'Turbidity': Turbidity
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

df_imported = pd.read_csv("Drinking_water.csv")
df2 = df_imported.dropna()
water = df2.sample(frac=1)


X = water.iloc[:, :-1]
Y = water.iloc[:, -1]

sc = StandardScaler()
X = sc.fit_transform(X)

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

Outcome = numpy.array(['Not potable', "Potable"])

st.subheader('Class labels and their corresponding index number')
st.write(Outcome)

st.subheader('The water is pridected as: ')
val = Outcome[prediction]
st.info(val)


st.subheader('Prediction Probability')
st.write(prediction_proba)
