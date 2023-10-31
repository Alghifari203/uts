import pickle
import streamlit as st

model = pickle.load(open('estimasi_housing.sav', 'rb'))

st.title('Stock Price Prediction App')
total_rooms = st.number_input('Input total rooms')
total_bedrooms = st.number_input('Input total bedrooms')
longitude = st.number_input('Input longitude ')
latitude = st.number_input('Input latitude')
population = st.number_input('Input population')
housing_median_age = st.number_input('Input housing median age')
households = st.number_input('Input households')
median_income = st.number_input('Input median income')
median_house_value = st.number_input('Input median house value')
ocean_proximity =  st.number_input('Input ocean proximity')
predict = ''

if st.button('predict'):
    predict = model.predict(
        [[total_rooms, total_bedrooms, longitude, latitude, population, housing_median_age, households, median_income, median_house_value, ocean_proximity ]]
    )
    st.write('population : ', predict)
