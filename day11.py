from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# ─── TOOLS ────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current information about any topic."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        output = ""
        for r in results:
            output += f"Title: {r['title']}\nSummary: {r['body']}\n\n"
        return output
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression like '100 * 0.15' or '75 * 6 * 20'."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_current_date(dummy: str = "") -> str:
    """Get today's date and day of the week."""
    now = datetime.datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Time: %H:%M")

@tool
def save_report(content: str) -> str:
    """Save text content as a report file."""
    try:
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Report saved as {filename}"
    except Exception as e:
        return f"Save failed: {str(e)}"

# ─── AGENT SETUP ──────────────────────────────────────────

tools = [web_search, calculate, get_current_date, save_report]

# InMemorySaver gives the agent memory across turns
checkpointer = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    prompt="""You are a helpful AI research assistant with access to tools.
Think step by step. Use tools when you need current information or calculations.
Always give a clear, structured final answer."""
)

# ─── RUN AGENT ────────────────────────────────────────────

def run_agent(task: str, thread_id: str = "default"):
    print("\n" + "=" * 60)
    print(f"🎯 TASK: {task}")
    print("=" * 60)

    config = {"configurable": {"thread_id": thread_id}}

    for step in agent.stream(
        {"messages": [{"role": "user", "content": task}]},
        config=config,
        stream_mode="updates"
    ):
        for node, values in step.items():
            messages = values.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n🔧 Using tool: {tc['name']}")
                        print(f"   Input: {tc['args']}")
                elif hasattr(msg, "content") and msg.content:
                    msg_type = type(msg).__name__
                    if "Tool" in msg_type:
                        print(f"   Result: {msg.content[:200]}...")
                    elif "AI" in msg_type:
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            print(f"\n✅ FINAL ANSWER:\n{msg.content}")

# ─── TEST TASKS ───────────────────────────────────────────

# Task 1 - simple tool
run_agent("What is today's date and what day of the week is it?", "task1")

# Task 2 - web search
run_agent(
    "Search for the top AI agent frameworks in 2026 and tell me which 3 are most popular.",
    "task2"
)

# Task 3 - multi-step calculation
run_agent(
    "A freelancer charges $75 per hour, works 6 hours a day for 20 working days a month. Calculate their monthly income in USD, then convert to Pakistani Rupees at 280 PKR per dollar.",
    "task3"
)
# Task 4 - save research report
run_agent(
    "Search for the top AI agent frameworks in 2026, summarize the findings, and save the report to a file.",
    "task4"
)