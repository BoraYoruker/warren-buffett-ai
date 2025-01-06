import os
import json
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import your helpers (they still contain get_stock_price, get_stock_news, etc.)
from helpers import get_stock_price, get_stock_news, extract_stock_symbol, is_valid_ticker

# -- Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env file or environment variables.")

# --------------------------
# 1) Define your function JSON specs for the LLM to call
# --------------------------
function_definitions = [
    {
        "name": "get_stock_price",
        "description": "Get the current price/fundamentals for a given stock symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. TSLA or AAPL"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_stock_news",
        "description": "Get the latest news headlines for the given stock symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. TSLA or AAPL"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "plot_stock_history",
        "description": "Generate a chart of the given stock symbol's historical data (e.g. TSLA). Note: In CLI mode, we just return a message that the chart was displayed.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. TSLA or AAPL"
                }
            },
            "required": ["symbol"]
        }
    }
]

# --------------------------
# 2) Implement the "plot_stock_history" function
#    (In CLI context, we'll just do fig.show() or return a message.)
# --------------------------
def plot_stock_history(symbol: str) -> str:
    """
    Since this is 'main.py' for CLI usage, we'll do the old fig.show() approach,
    then return a message. If you want to purely return a figure, do that in app.py.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period='1mo', interval='1d')
    if data.empty:
        return f"No historical data found for {symbol}."
    # Create a simple plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.update_layout(
        title=f"{symbol} Stock History",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    fig.show()
    return f"Chart for {symbol} has been displayed."

# --------------------------
# 3) RAG Setup: Build or load the vector store
# --------------------------
def chunk_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error reading or splitting file {file_path}: {e}")
        return []

def get_vector_store(file_list, recreate=False):
    vector_store_path = "vector_store.faiss"
    embedding_model = OpenAIEmbeddings()

    if not recreate and os.path.exists(vector_store_path):
        print("Loading existing vector store...")
        return FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

    print("Creating new vector store...")
    chunks = []
    for file in file_list:
        chunks.extend(chunk_text(file))

    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local(vector_store_path)
    return vector_store

# --------------------------
# 4) The LLM: function-calling approach
# --------------------------
def call_llm_with_functions(user_prompt: str, vector_store) -> str:
    """
    Let the LLM decide whether to call get_stock_price, get_stock_news, or plot_stock_history,
    or just answer from RAG. We'll handle function calls if the LLM chooses to do so.
    """
    # Step A: Make RAG context
    docs = vector_store.similarity_search(user_prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step B: Create system+user messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are Warren Buffett, a legendary investor known for your value investing philosophy. "
                "Use the 'functions' below if relevant: get_stock_price, get_stock_news, plot_stock_history. "
                "If the user asks about a stock's price or fundamentals, call get_stock_price. If they want news, "
                "call get_stock_news. If they want a chart or graph, call plot_stock_history. If they ask a more "
                "general question, incorporate context from the vector store. The user might have typos or might say "
                "'tesla' or 'tsla'—do your best to interpret. If no function is needed, just answer with your knowledge."
            )
        },
        {
            "role": "system",
            "content": f"Relevant Context from Documents:\n{context}"
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        model_kwargs={
            "functions": function_definitions,
            "function_call": "auto"
        }
    )

    response = chat.invoke(messages)
    # Step C: If the LLM calls a function, handle it
    function_call = response.additional_kwargs.get("function_call")
    if function_call:
        fn_name = function_call["name"]
        try:
            args = json.loads(function_call["arguments"])
        except:
            return "Invalid JSON in function arguments."

        # We interpret the function name
        if fn_name == "get_stock_price":
            symbol = args.get("symbol", "")
            return get_stock_price(symbol)

        elif fn_name == "get_stock_news":
            symbol = args.get("symbol", "")
            return get_stock_news(symbol)

        elif fn_name == "plot_stock_history":
            symbol = args.get("symbol", "")
            return plot_stock_history(symbol)
        else:
            return f"Function {fn_name} not implemented."

    else:
        # No function call -> just return the LLM's text
        return response.content

# --------------------------
# 5) Command-Line Interface
# --------------------------
def main():
    print("=== Warren Buffett Stock Advisor ===\n")

    data_files = [
        "ALL_Letters.txt",
        "ESSAYS_WARREN.txt",
        "ANNUAL_MEETING_TRANSCRIPTS.txt"
    ]
    vector_store = get_vector_store(data_files, recreate=False)

    while True:
        user_input = input("Ask Warren Buffett (type 'exit' to quit): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = call_llm_with_functions(user_input, vector_store)
        print(f"\nWarren Buffett Says:\n{answer}\n")

if __name__ == "__main__":
    main()
