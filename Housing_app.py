import pickle
import streamlit as st

model = pickle.load(open('estimasi_housing.sav', 'rb'))

st.title('Stock Price Prediction App')
total_rooms = st.number_input('Input total rooms')
total_bedrooms = st.number_input('Input total bedrooms')
longitude = st.number_input('Input longitude ')

predict = ''

if st.button('predict'):
    predict = model.predict(
        [[total_rooms, total_bedrooms, longitude]]
    )
    st.write('population : ', predict)
