# Stock Market Predictor using LSTM
![image](https://github.com/kinjal30/Stock-Prediction-using-Keras-LSTM/assets/46399913/616f8037-047b-4856-9ff2-fb551b1e4799)




This repository contains a Streamlit web application (`app.py`) that utilizes a trained Long Short-Term Memory (LSTM) neural network model to predict stock prices. The model is trained on historical stock price data fetched from Yahoo Finance using the `yfinance` library.

## Dependencies

The following Python libraries are required to run the application:

- `numpy`
- `pandas`
- `yfinance`
- `keras`
- `streamlit`
- `matplotlib`

You can install these dependencies using `pip install <package_name>`.

## Usage

To use the application:

1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Run the Streamlit app using the command `streamlit run app.py`.
4. Enter the stock symbol in the input field. By default, the symbol is set to `GOOG` (Google).
5. The application will fetch historical stock price data and display it along with moving averages.
6. It will then use the trained LSTM model to predict future stock prices and display the predictions alongside the actual prices.

## Model Loading

The pre-trained LSTM model (`Stock Prediction model.h5`) is loaded using the `load_model` function from the Keras library.

## Data Preprocessing

The historical stock price data is preprocessed by splitting it into training and testing sets, and then scaling it using MinMaxScaler.

## Visualization

The application visualizes the following:

- Stock price data along with a 50-day moving average.
- Stock price data along with 50-day and 100-day moving averages.
- Stock price data along with 100-day and 200-day moving averages.
- Original and predicted stock prices.

Feel free to explore different stock symbols, date ranges, and model architectures to analyze and predict stock prices.
