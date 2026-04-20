import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Train the model
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Website UI
st.title("🏠 House Price Predictor")
st.write("Enter house details below to get a predicted price:")

med_inc     = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 5.0)
house_age   = st.slider("House Age (years)", 1, 52, 20)
ave_rooms   = st.slider("Average Rooms", 1.0, 10.0, 5.0)
ave_bedrms  = st.slider("Average Bedrooms", 1.0, 5.0, 1.0)
population  = st.slider("Neighborhood Population", 100, 5000, 1000)
ave_occup   = st.slider("Average Occupants per House", 1.0, 6.0, 2.5)
latitude    = st.slider("Latitude", 32.0, 42.0, 37.0)
longitude   = st.slider("Longitude", -125.0, -114.0, -120.0)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms,
                                  population, ave_occup, latitude, longitude]],
                               columns=data.feature_names)
    predicted = model.predict(input_data)
    st.success(f"💰 Predicted House Price: ${predicted[0]*100000:,.0f}")