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
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# ─── TOOLS ────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for information about a company or topic."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found."
        output = ""
        for r in results:
            output += f"Title: {r['title']}\nSummary: {r['body']}\nURL: {r['href']}\n\n"
        return output
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def analyze_business(company_info: str) -> str:
    """Analyze a business description and identify AI automation opportunities."""
    opportunities = []
    keywords = {
        "customer support": "AI chatbot to handle customer queries 24/7",
        "e-commerce": "AI product recommendations and inventory management",
        "content": "AI content generation and SEO optimization pipeline",
        "real estate": "AI lead qualification and property matching agent",
        "restaurant": "AI reservation system and menu recommendation bot",
        "healthcare": "AI appointment scheduling and patient FAQ bot",
        "finance": "AI document analysis and report generation",
        "education": "AI personalized tutoring and progress tracking",
        "marketing": "AI campaign optimization and audience targeting",
        "logistics": "AI route optimization and delivery tracking",
        "delivery": "AI route optimization and delivery tracking",
        "fashion": "AI product recommendations and inventory management",
        "hr": "AI resume screening and candidate matching",
        "legal": "AI contract analysis and document summarization"
    }
    company_lower = company_info.lower()
    for keyword, opportunity in keywords.items():
        if keyword in company_lower:
            if opportunity not in opportunities:
                opportunities.append(f"• {opportunity}")

    if not opportunities:
        opportunities = [
            "• AI automation of repetitive workflows",
            "• AI chatbot for customer engagement",
            "• AI data analysis and reporting"
        ]
    return "Identified AI opportunities:\n" + "\n".join(opportunities[:3])

@tool
def save_proposal(company_name: str, proposal_text: str) -> str:
    """Save the final proposal to a file. 
    Args:
        company_name: Clean company name for the filename e.g. 'Daraz Pakistan'
        proposal_text: The COMPLETE full proposal text to save. Must be the entire proposal, not a summary or filename.
    """
    try:
        # Validate that actual proposal content was passed
        if len(proposal_text) < 200:
            return "ERROR: proposal_text is too short. Pass the complete proposal text, not a summary or filename."
        
        os.makedirs("proposals", exist_ok=True)
        clean_name = company_name.replace(" ", "_").replace("/", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"proposals/{clean_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Company: {company_name}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(proposal_text)
        
        return f"✅ Proposal saved to {filename} ({len(proposal_text)} characters)"
    except Exception as e:
        return f"Save failed: {str(e)}"

# ─── AGENT ────────────────────────────────────────────────

tools = [web_search, analyze_business, save_proposal]
checkpointer = InMemorySaver()

SYSTEM_PROMPT = """You are an expert AI automation consultant and business researcher.

Your job is to research a company and write a personalized outreach proposal offering AI automation services.

Follow these EXACT steps for every company:
1. Use web_search to find what the company does, their size, and recent news
2. Use web_search again to find their specific challenges or growth plans
3. Use analyze_business with a detailed description of what they do
4. Write a complete proposal (250-300 words) with:
   - Personal opening referencing specific facts you found
   - 3 specific AI automation opportunities for THEIR situation with ROI estimates
   - Clear call to action
5. Use save_proposal tool with TWO arguments:
   - company_name: the company name
   - proposal_text: THE COMPLETE FULL PROPOSAL TEXT YOU JUST WROTE — every word of it, not a summary, not a filename, the actual proposal

CRITICAL: When calling save_proposal, the proposal_text argument must contain the entire proposal letter from "Dear..." to your sign-off. Never pass a filename or summary."""

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    prompt=SYSTEM_PROMPT
)

# ─── RUN ──────────────────────────────────────────────────

def research_and_propose(company: str):
    print("\n" + "=" * 60)
    print(f"🔍 Researching: {company}")
    print("=" * 60)

    config = {"configurable": {"thread_id": company}}
    final_proposal = ""

    for step in agent.stream(
        {"messages": [{"role": "user", "content": f"Research this company and write a complete AI automation proposal, then save it: {company}"}]},
        config=config,
        stream_mode="updates"
    ):
        for node, values in step.items():
            messages = values.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n🔧 {tc['name']}")
                        if tc['name'] != 'save_proposal':
                            print(f"   Input: {str(tc['args'])[:120]}")
                elif hasattr(msg, "content") and msg.content:
                    msg_type = type(msg).__name__
                    if "Tool" in msg_type:
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        print(f"   ↳ {content[:300]}")
                    elif "AI" in msg_type:
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if content.strip():
                                final_proposal = content
                                print(f"\n✅ COMPLETE:\n{content}")

    print("\n" + "-" * 60)

# Test with 3 companies
research_and_propose("Sapphire Pakistan clothing brand")
research_and_propose("Foodpanda Pakistan food delivery")
research_and_propose("Daraz Pakistan e-commerce")