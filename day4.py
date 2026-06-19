from google import genai
try:
    from dotenv import load_dotenv
except Exception:
    # If python-dotenv is not installed, provide a no-op fallback
    def load_dotenv():
        return
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def scrape_website(url):
    """Read text content from any website"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        # Get clean text
        text = soup.get_text(separator=" ", strip=True)
        
        # Limit to first 2000 characters (enough for AI to understand the business)
        return text[:2000]
    
    except Exception as e:
        return f"Could not read website: {str(e)}"

def generate_proposal(business_url, business_text):
    """Generate a professional AI services proposal"""
    
    prompt = f"""You are an expert AI automation consultant writing a cold outreach proposal.

A potential client's website content is below. Analyze their business and write a 
professional, personalized proposal offering AI automation services.

Website: {business_url}
Business Content: {business_text}

Write a proposal that includes:
1. A personalized opening showing you understood their business (2-3 sentences)
2. 3 specific AI automation opportunities for THEIR business (be specific, not generic)
3. Expected benefits (time saved, cost reduced, revenue increased)
4. A simple call to action

Tone: Professional but conversational. Not salesy. Like one professional talking to another.
Length: 250-300 words maximum.
Format: Plain text, no markdown symbols."""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text

def analyze_business(url):
    print(f"\n🔍 Reading website: {url}")
    business_text = scrape_website(url)
    
    if "Could not read" in business_text:
        print(f"❌ Error: {business_text}")
        return
    
    print("✅ Website read successfully")
    print("🤖 Generating proposal...")
    
    proposal = generate_proposal(url, business_text)
    
    print("\n" + "=" * 60)
    print("YOUR AI-GENERATED PROPOSAL")
    print("=" * 60)
    print(proposal)
    print("=" * 60)
    
    # Save to file
    filename = f"proposal_{url.replace('https://','').replace('http://','').replace('/','_')[:30]}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Proposal for: {url}\n\n")
        f.write(proposal)
    
    print(f"\n💾 Saved to: {filename}")

# Test with 3 real businesses
websites = [
    "https://www.fiverr.com",
    "https://automattic.com",
    "https://pk.sapphireonline.pk/"
]

for site in websites:
    analyze_business(site)
    print("\n")