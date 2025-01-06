import requests
import yfinance as yf
import re
import os
from datetime import datetime, timedelta

def get_stock_news(stock_symbol):
    api_key = os.getenv('NEWS_API_KEY')  # Load the API key from the environment variable

    if not api_key:
        return "API key not found. Please set the NEWS_API_KEY environment variable."

    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        news = [f"Title: {article['title']}\nDescription: {article['description']}" for article in articles[:3]]
        return "\n\n".join(news) if news else "No news found."
    else:
        return f"Error fetching news: {response.status_code} - {response.text}"


def get_stock_price(stock_symbol):
    """
    Fetches the latest stock price for a given symbol using Yahoo Finance's history data.
    Handles cases where the market is closed by falling back to the previous closing price.
    """
    try:
        symbol = stock_symbol.strip().upper()
        print(f"Fetching stock price for: {symbol}")  # Debugging statement
        stock = yf.Ticker(symbol)

        # Fetch the last 5 days of history to account for weekends/holidays
        hist = stock.history(period="5d", interval="1d")
        print(f"Historical data fetched:\n{hist}")  # Debugging statement

        if hist.empty:
            return f"Could not retrieve data for {symbol}. Please ensure the ticker symbol is correct."

        # Get the last available date's closing price
        latest_date = hist.index.max()
        latest_close = hist.loc[latest_date]['Close']
        price_date = latest_date.strftime('%Y-%m-%d')

        currency = "USD"  # Assuming USD; adjust if necessary

        # Fetch additional info if needed
        info = stock.info
        market_cap = info.get('marketCap')
        forward_pe = info.get('forwardPE')
        dividend_yield = info.get('dividendYield')

        # Build a reply string
        reply = f"The closing price of {symbol} on {price_date} was ${latest_close:,.2f} {currency}."
        if market_cap:
            reply += f" Market Cap: ${market_cap:,.0f}."
        if forward_pe:
            reply += f" Forward P/E: {forward_pe:.2f}."
        if dividend_yield:
            reply += f" Dividend Yield: {dividend_yield:.2%}."

        return reply.strip()
    except Exception as e:
        print(f"Error in get_stock_price: {e}")  # Debugging statement
        return f"Error fetching stock data for {stock_symbol}: {e}"


def is_valid_ticker(symbol: str) -> bool:
    """
    Quick check using yfinance. Returns True if the given symbol
    appears valid (i.e., we can fetch at least some history).
    Returns False otherwise.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        # If hist is empty, that usually means invalid ticker
        if hist.empty:
            return False
        return True
    except:
        return False


def extract_stock_symbol(prompt: str) -> str or None:
    """
    Extracts a potential ticker symbol (1-5 letters) from the user's input
    and checks if it's valid using yfinance. Returns the valid symbol
    if found, else None.
    """
    # Basic regex for something that might be a ticker (1 to 5 letters)
    match = re.search(r'\b([A-Za-z]{1,5})\b', prompt.upper().strip())
    if match:
        symbol_candidate = match.group(1).upper()
        if is_valid_ticker(symbol_candidate):
            return symbol_candidate

    return None
