
"""
Simple inference script for ARIMA stock price prediction
"""
import pickle
import pandas as pd
import yfinance as yf

def load_model(model_path='best_arima_model.pkl'):
    """Load the trained ARIMA model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_next_day(model, ticker='AAPL', days_history=100):
    """
    Predict next day's stock price
    
    Args:
        model: Trained ARIMA model
        ticker: Stock ticker symbol
        days_history: Number of historical days to use
    
    Returns:
        Dictionary with prediction results
    """
    # Download recent data
    data = yf.download(ticker, period=f'{days_history}d', progress=False)
    
    # Make prediction
    forecast = model.forecast(steps=1)
    current_price = data['Close'].iloc[-1]
    predicted_price = forecast.values[0]
    
    return {
        'ticker': ticker,
        'current_price': float(current_price),
        'predicted_price': float(predicted_price),
        'change': float(predicted_price - current_price),
        'change_percent': float((predicted_price - current_price) / current_price * 100)
    }

if __name__ == "__main__":
    # Example usage
    print("Loading ARIMA model...")
    model = load_model()
    
    print("Making prediction for AAPL...")
    result = predict_next_day(model)
    
    print(f"\nPrediction Results:")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Predicted Price: ${result['predicted_price']:.2f}")
    print(f"Expected Change: ${result['change']:.2f} ({result['change_percent']:+.2f}%)")