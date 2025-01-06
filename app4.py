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

# Function to call the LLM with memory
def call_llm_with_memory(user_prompt, vector_store, memory):
    """
    Handle conversation with the LLM, supporting function calls and memory.
    """
    # Use vector store for RAG context
    docs = vector_store.similarity_search(user_prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Compose messages with context and full memory
    messages = memory + [
        {
            "role": "system",
            "content": (
                "You are Warren Buffett, a legendary investor. You can call these functions if needed: "
                "get_stock_price, get_stock_news, plot_stock_history. Interpret user input with possible typos. "
                "If no function is relevant, just respond in text."
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Add relevant context from vector store as a system message
    messages.insert(1, {"role": "system", "content": f"Relevant Context:\n{context}"})

    # Call the LLM with function-calling enabled
    chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        functions=function_definitions,
        function_call="auto"
    )
    response = chat.invoke(messages)

    # Handle function calls
    function_call = response.additional_kwargs.get("function_call")
    if function_call:
        fn_name = function_call["name"]
        args = json.loads(function_call["arguments"])

        if fn_name == "get_stock_price":
            return "assistant", get_stock_price(args["symbol"]), None
        elif fn_name == "get_stock_news":
            return "assistant", get_stock_news(args["symbol"]), None
        elif fn_name == "plot_stock_history":
            fig = plot_stock_history(args["symbol"])
            if not fig:
                return "assistant", f"No historical data found for {args['symbol']}.", None
            return "assistant", f"Here is the chart for {args['symbol']}.", fig

    # Handle regular text responses
    return "assistant", response.content, None

# Streamlit app function
# ... [Previous imports and code]

# Streamlit app function
def main():
    st.title("Warren Buffett AI")

    # Add Reset Conversation button in the sidebar
    with st.sidebar:
        st.header("Controls")
        if st.button("Reset Conversation"):
            st.session_state["messages"] = [{"role": "system", "content": "You are Warren Buffett, a legendary investor."}]
            st.success("Conversation has been reset.")

    # Load vector store for RAG
    data_files = ["ALL_Letters.txt", "ESSAYS_WARREN.txt", "ANNUAL_MEETING_TRANSCRIPTS.txt"]
    vector_store = get_vector_store(data_files, recreate=False)

    # Initialize conversation memory in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are Warren Buffett, a legendary investor."}]

    # Display conversation history
    for i, msg in enumerate(st.session_state["messages"]):
        # Skip displaying the system message
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Warren Buffett:** {msg['content']}")
            if msg.get("figure"):
                # Use a unique key for each chart
                st.plotly_chart(msg["figure"], use_container_width=True, key=f"chart-{i}")

    # Handle user input and processing
    def on_send():
        user_msg = st.session_state["user_input"].strip()
        if not user_msg:
            st.warning("Please enter a valid query.")
            return  # Ignore empty input

        # Add user message to memory
        st.session_state["user_input"] = ""
        st.session_state["messages"].append({"role": "user", "content": user_msg})

        # Call the LLM with memory
        role, content, figure = call_llm_with_memory(user_msg, vector_store, st.session_state["messages"])
        st.session_state["messages"].append({"role": role, "content": content, "figure": figure})

    # Input box for user queries
    st.text_input(
        "Ask Warren Buffett a question:",
        key="user_input",
        on_change=on_send,
        placeholder="e.g., Tesla price, Graph NVIDIA, News about AAPL, etc."
    )

# ... [Rest of the code remains unchanged]

if __name__ == "__main__":
    main()
