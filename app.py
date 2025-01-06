import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from main import get_vector_store  # We'll still build/load the vector store
from helpers import get_stock_price, get_stock_news
import yfinance as yf
import plotly.graph_objects as go

# 1) Load environment vars
load_dotenv()

# 2) Define function specs for function calling
function_definitions = [
    {
        "name": "get_stock_price",
        "description": "Get the current price/fundamentals for a given stock symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. TSLA or AAPL"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_stock_news",
        "description": "Get the latest news for the given stock symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. TSLA or AAPL"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "plot_stock_history",
        "description": "Generate a Plotly figure for the given stock symbol's historical data.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. TSLA or AAPL"
                }
            },
            "required": ["symbol"]
        }
    }
]

def plot_stock_history(symbol: str):
    """
    Return a Plotly figure of the stock's historical data for Streamlit.
    """
    def get_stock_history(sym, period='1mo', interval='1d'):
        stock = yf.Ticker(sym)
        return stock.history(period=period, interval=interval)

    periods = {
        '1d':  get_stock_history(symbol, '1d', '1m'),
        '5d':  get_stock_history(symbol, '5d', '5m'),
        '1m':  get_stock_history(symbol, '1mo', '1h'),
        '6m':  get_stock_history(symbol, '6mo', '1d'),
        'YTD': get_stock_history(symbol, 'ytd', '1d'),
        '1y':  get_stock_history(symbol, '1y', '1d'),
        '5y':  get_stock_history(symbol, '5y', '1wk')
    }

    default_period = '1m'
    if periods[default_period].empty:
        return None

    fig = go.Figure()
    period_names = list(periods.keys())

    for i, period_name in enumerate(period_names):
        df = periods[period_name]
        visible = (period_name == default_period)
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name=period_name,
                    visible=visible
                )
            )

    buttons = []
    for i, period_name in enumerate(period_names):
        buttons.append(
            dict(
                label=period_name,
                method='update',
                args=[
                    {'visible': [j == i for j in range(len(periods))]},
                    {'title': f"{symbol} Stock Price History - {period_name}"}
                ]
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=2,
                buttons=buttons,
                direction="down",
                showactive=True
            )
        ],
        title=f"{symbol} Stock Price History - {default_period}",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    return fig

def call_llm_with_functions(user_prompt: str, vector_store):
    """
    In Streamlit, we won't do print() or input().
    We'll just return either normal text or instructions to display a figure.
    """
    # Let's do minimal RAG: get context from vector store
    docs = vector_store.similarity_search(user_prompt, k=3)
    context_text = "\n\n".join([d.page_content for d in docs])

    # Compose messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are Warren Buffett, a legendary investor. You can call these functions if needed: "
                "get_stock_price, get_stock_news, plot_stock_history. Interpret user input with possible typos. "
                "If no function is relevant, just respond in text."
            )
        },
        {
            "role": "system",
            "content": f"Relevant context:\n{context_text}"
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    chat = ChatOpenAI(
        model="gpt-4o",  # or gpt-4-0613
        temperature=0,
        functions=function_definitions,
        function_call="auto"
    )

    response = chat.invoke(messages)
    function_call = response.additional_kwargs.get("function_call")
    if function_call:
        fn_name = function_call["name"]
        try:
            args = json.loads(function_call["arguments"])
        except:
            return ("assistant", "I tried to call a function but the arguments were invalid JSON.", None)

        symbol = args.get("symbol", "").upper().strip()
        if fn_name == "get_stock_price":
            return ("assistant", get_stock_price(symbol), None)
        elif fn_name == "get_stock_news":
            return ("assistant", get_stock_news(symbol), None)
        elif fn_name == "plot_stock_history":
            # Return a figure
            fig = plot_stock_history(symbol)
            if not fig:
                return ("assistant", f"No historical data found for {symbol}.", None)
            return ("assistant", f"Here is the chart for {symbol}.", fig)
        else:
            return ("assistant", f"Function {fn_name} not implemented.", None)
    else:
        # Just normal text response
        return ("assistant", response.content, None)

def main():
    st.title("Warren Buffett AI Stock Advisor")

    data_files = [
        "ALL_Letters.txt",
        "ESSAYS_WARREN.txt",
        "ANNUAL_MEETING_TRANSCRIPTS.txt"
    ]
    vector_store = get_vector_store(data_files, recreate=False)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.subheader("Conversation")
    if not st.session_state["messages"]:
        st.write("_No messages yet. Ask a question below._")
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Warren Buffett:** {msg['content']}")
                if msg.get("figure"):
                    st.plotly_chart(msg["figure"], use_container_width=True)

    def on_send():
        user_msg = st.session_state["user_input"].strip()
        if not user_msg:
            return
        st.session_state["user_input"] = ""

        # Add user message
        st.session_state["messages"].append(
            {"role": "user", "content": user_msg}
        )

        # Call the LLM with function calling
        role, content, figure = call_llm_with_functions(user_msg, vector_store)
        st.session_state["messages"].append(
            {"role": role, "content": content, "figure": figure}
        )

    st.text_input(
        label="Ask Warren Buffett a question:",
        key="user_input",
        on_change=on_send,
        placeholder="e.g. Price of tesla, show me chart of msft, etc."
    )

if __name__ == "__main__":
    main()
