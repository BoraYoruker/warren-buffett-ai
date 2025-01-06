import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from helpers import get_stock_price, get_stock_news, extract_stock_symbol, is_valid_ticker
import yfinance as yf
import plotly.graph_objects as go
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from main import get_vector_store  # Ensure this import path is correct

# 1) Load environment variables
load_dotenv()

# 2) Define function specifications for function calling
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

# Define maximum number of messages to store
MAX_MESSAGES = 50
MAX_MESSAGES_BEFORE_SUMMARY = 30

# Function to summarize past messages
def summarize_messages(messages):
    """
    Summarize the conversation history to maintain context without exceeding message limits.

    Args:
        messages (list): List of message dictionaries.

    Returns:
        list: Summarized messages list.
    """
    # Extract messages to summarize (e.g., first 20 messages)
    messages_to_summarize = messages[:20]
    remaining_messages = messages[20:]

    # Create a prompt for summarization
    summary_prompt = "Summarize the following conversation between a user and Warren Buffett, focusing on key topics and information exchanged:\n\n"
    for msg in messages_to_summarize:
        role = "User" if msg["role"] == "user" else "Warren Buffett"
        summary_prompt += f"{role}: {msg['content']}\n"

    # Define summarization function
    summary_function_definition = {
        "name": "summarize_conversation",
        "description": "Summarize the conversation history.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A concise summary of the conversation."
                }
            },
            "required": ["summary"]
        }
    }

    # Call the LLM to generate summary
    chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        model_kwargs={
            "functions": function_definitions,
            "function_call": "auto"
        }
    )
    response = chat.invoke([
        {
            "role": "system",
            "content": "You are Warren Buffett, a legendary investor who can summarize conversations."
        },
        {
            "role": "user",
            "content": summary_prompt
        }
    ])

    # Handle function call for summarization
    function_call = response.additional_kwargs.get("function_call")
    if function_call and function_call["name"] == "summarize_conversation":
        summary = json.loads(function_call["arguments"])["summary"]
        # Create a summarized system message
        summarized_message = {"role": "system", "content": f"Summary of previous conversation:\n{summary}"}
        # Combine summarized message with remaining messages
        return [summarized_message] + remaining_messages
    else:
        # If summarization fails, return original messages
        return messages



# (ADDED) A new chain-of-thought message that is used internally only.
CHAIN_OF_THOUGHT_MESSAGE = {
    "role": "system",
    "content": (
        "You are about to receive a user question and a set of possible function calls. "
        "Please reason step by step (chain-of-thought) and figure out the best possible answer. "
        "You may call the available functions if you need data such as stock price, news, or historical charts. "
        "However, do NOT reveal your full chain-of-thought to the user. "
        "Instead, provide a concise final response that summarizes your reasoning without exposing the entire chain-of-thought."
    )
}

def call_llm_with_memory(user_prompt, vector_store, memory):
    """
    Handle conversation with the LLM, supporting function calls, memory, and chain-of-thought reasoning.
    """
    # 1) Retrieve relevant docs from the vector store (RAG)
    docs = vector_store.similarity_search(user_prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2) Build refined system message
    system_message = (
        "You are Warren Buffett, the legendary investor. You have an extensive history of analyzing companies, "
        "giving direct opinions, and taking strong stances on investment decisions. When the user asks a question "
        "like 'Should I buy AMD?' or 'Is it a good time to sell Tesla?', you will provide a clear and direct recommendation "
        "as though you are Warren Buffett. Incorporate relevant market data (price, history, fundamentals, news) and your own "
        "value-investing philosophy.\n\n"
        "Answer questions with a clear buy/hold/sell recommendation (e.g. 'I recommend you buy AMD right now...' or 'Currently you should not buy AMD...'), "
        "along with your detailed reasoning. Use the user’s question, the context from the vector store, and the current stock data. "
        "Feel free to reference your real-world investing principles (e.g. focusing on a company's moat, stability, etc.). "
        "Additionally, it’s okay to express personal convictions. "
        "\n\n"
        "Available functions: get_stock_price(symbol), get_stock_news(symbol), plot_stock_history(symbol). "
        "Use them if needed to retrieve or display data, but always come back with a clear recommendation. (call the 'plot_stock_history(symbol)' most often. "
    )

    # 3) Compose the full list of messages to send to the LLM.
    messages_for_llm = memory + [
        CHAIN_OF_THOUGHT_MESSAGE,  # Chain-of-thought instruction
        {"role": "system", "content": system_message},
        {
            "role": "system",
            "content": (
                f"Relevant Context (use this in your response if helpful):\n{context}\n\n"
                "You must integrate this context if it’s relevant. Quote or paraphrase from it to support your recommendation."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    max_iterations = 3
    iteration = 0
    final_response = None
    final_figure = None

    # Create an instance of ChatOpenAI with function-calling
    chat = ChatOpenAI(
        model="gpt-4",  # Corrected model name if necessary
        temperature=0.7,
        model_kwargs={
            "functions": function_definitions,
            "function_call": "auto"
        }
    )

    while iteration < max_iterations:
        iteration += 1
        response = chat.invoke(messages_for_llm)  # Corrected variable name
        function_call = response.additional_kwargs.get("function_call")

        if function_call:
            # The LLM decided it needs a function call
            fn_name = function_call["name"]
            args = json.loads(function_call["arguments"] or "{}")

            # Execute the requested function
            if fn_name == "get_stock_price":
                function_result = get_stock_price(args["symbol"])
                final_figure = None
            elif fn_name == "get_stock_news":
                function_result = get_stock_news(args["symbol"])
                final_figure = None
            elif fn_name == "plot_stock_history":
                fig = plot_stock_history(args["symbol"])
                if not fig:
                    function_result = f"No historical data found for {args['symbol']}."
                    final_figure = None
                else:
                    function_result = f"Here is the chart for {args['symbol']}."
                    final_figure = fig
            else:
                function_result = f"Function '{fn_name}' not recognized."
                final_figure = None

            # Add the function result as a message
            messages_for_llm.append({
                "role": "function",
                "name": fn_name,
                "content": function_result
            })

            # Add a small "system" reminder to produce the final direct recommendation
            messages_for_llm.append({
                "role": "system",
                "content": (
                    "You have just received the function result above. "
                    "Use your chain-of-thought to incorporate this data into your final recommendation. "
                    "Do NOT repeat the entire chain-of-thought. Provide a single short answer that addresses the user."
                )
            })

            # The loop will continue for the next iteration to incorporate the function result
        else:
            # The model returned a direct text response (no function call).
            final_response = response.content
            break

    if final_response is None:
        # If we exit the loop without a direct text response, fallback:
        final_response = "I’m sorry, I couldn’t formulate a final answer."

    return "assistant", final_response, final_figure



def main():
    st.title("Warren Buffet AI")

    # Add Reset Conversation button in the sidebar
    with st.sidebar:
        st.header("Controls")
        if st.button("Reset Conversation"):
            st.session_state["messages"] = [
                {"role": "system", "content": "You are Warren Buffett, a legendary investor."}]
            st.success("Conversation has been reset.")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

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
            # Use st.text to prevent markdown from interpreting line breaks
            st.text(f"Warren Buffett: {msg['content']}")
            if msg.get("figure"):
                # Use a unique key for each chart
                st.plotly_chart(msg["figure"], use_container_width=True, key=f"chart-{i}")

    # Handle user input and processing
    def on_send():
        user_msg = st.session_state["user_input"].strip()
        if not user_msg:
            st.warning("Please enter a valid query.")
            return  # Ignore empty input

        # Extract stock symbol to validate
        symbol = extract_stock_symbol(user_msg)
        if symbol and not is_valid_ticker(symbol):
            st.warning(f"'{symbol}' is not a valid ticker symbol. Please try again.")
            return

        # Add user message to memory
        st.session_state["user_input"] = ""
        st.session_state["messages"].append({"role": "user", "content": user_msg})

        # Display loading indicator while processing
        with st.spinner("Processing your request..."):
            # Call the LLM with memory
            role, content, figure = call_llm_with_memory(user_msg, vector_store, st.session_state["messages"])

        # Append assistant's response to messages
        st.session_state["messages"].append({"role": role, "content": content, "figure": figure})

        # Ensure the message list does not exceed MAX_MESSAGES
        if len(st.session_state["messages"]) > MAX_MESSAGES:
            st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

    # Input box for user queries
    st.text_input(
        "Ask Warren Buffett a question:",
        key="user_input",
        on_change=on_send,
        placeholder="e.g., Tesla price, Graph NVIDIA, News about AAPL, etc."
    )

if __name__ == "__main__":
    main()