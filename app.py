import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'C:\Users\keshr\Downloads\Stock\Stock Prediction model.h5')
st.header('Stock Market Predictor')

stock=st.text_input('Enter Stock Symbol', 'GOOG')
start = '2013-01-01'
end = '2023-1-31'

data = yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)

train_data = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
test_data = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = train_data.tail(100)
test_data = pd.concat([past_100_days, test_data], ignore_index=True)
test_data_scale=scaler.fit_transform(test_data)

st.subheader('MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'b')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'g')
plt.plot(data.Close,'b')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'g')
plt.plot(data.Close,'b')
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100, test_data_scale.shape[0]):
    x.append(test_data_scale[i-100:i])
    y.append(test_data_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict*scale
y = y*scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,'r', label='Original Price')
plt.plot(y,'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)