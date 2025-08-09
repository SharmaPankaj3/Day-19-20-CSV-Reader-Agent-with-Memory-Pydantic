# # Day 19-20: Agent with Memory + CSV Tool + Web Search + Pydantic
# pip install langchain
# pip install openai tiktoken pydantic
# pip install langchain-openai
# pip install duckduckgo-search
# pip install langchain langchain-openai pandas


from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
import pandas as pd
import os


#  load API key
with open(r"D:\desktop\Key_GEN_AI.txt","r") as f:
     os.environ["OPENAI_API_KEY"] = f.read().strip()

# "D:\desktop\ML\Analytics\Pankaj Analytics\ANA DATA\CardioGoodFitness.csv"

# ========= LLM =========
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ========= Memory =========
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ========= Global CSV Cache =========
df_cache = {"df": None}

# ========= Tool 1: Web Search =========
class SearchInput(BaseModel):
    query: str = Field(..., description="Search query for the web.")

def search_ddg(query: str) -> str:
    output = ""
    with DDGS() as ddgs:
        results = ddgs.text(query)
        for i, r in enumerate(results):
            if i >= 3:
                break
            output += f"{r['title']} ({r['href']})\n"
    return output or "No results found."

search_tool = Tool(
    name="DuckDuckGoSearch",
    func=search_ddg,
    description="Search the web using DuckDuckGO. Input should be a dictionary with key 'query'."
)

# ========= Tool 2: CSV Loader & Query =========
class CSVInput(BaseModel):
    filepath: str = Field(..., description="Path to the CSV file.")

def load_csv(filepath: str) -> str:
    try:
        df_cache["df"] = pd.read_csv(filepath)
        return f"CSV loaded successfully with {df_cache['df'].shape[0]} rows and {df_cache['df'].shape[1]} columns."
    except Exception as e:
        return f"Error loading CSV: {str(e)}"

csv_tool = Tool(
    name="LoadCSVFile",
    func=lambda input: load_csv(**eval(input)),
    description="Load a CSV file from a given path. Input must be a dictionary with key 'filepath'."
)

# ========= Tool 3: Ask CSV Questions =========
class AskCSVInput(BaseModel):
    question: str = Field(..., description="Question about the loaded CSV.")

def ask_csv(question: str) -> str:
    if df_cache["df"] is None:
        return "No CSV loaded yet. Please load one first."
    try:
        df = df_cache["df"]
        return str(df.head(3)) if "show" in question.lower() else str(df.describe())
    except Exception as e:
        return f"Error querying CSV: {str(e)}"

ask_csv_tool = Tool(
    name="AskCSV",
    func=lambda input: ask_csv(**eval(input)),
    description="Ask questions about the loaded CSV. Input must be a dictionary with key 'question'."
)


# ========= Agent =========
agent = initialize_agent(
    tools=[search_tool, csv_tool, ask_csv_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# ========= Example Conversation =========
print(agent.invoke({"input": "Hi, I am Pankaj."}))

print(agent.invoke({"input": "Load a CSV file from this path: {'filepath':'D:\desktop\ML\Analytics\Pankaj Analytics\ANA DATA\CardioGoodFitness.csv'}"}))

print(agent.invoke({"input": "Show me the first few rows"}))

print(agent.invoke({"input": "Search AI news in India"}))

print(agent.invoke({"input": "give the column names from the loaded csv."}))



