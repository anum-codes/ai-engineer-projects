from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

app = FastAPI(
    title="AI Automation API",
    description="REST API for AI-powered business tools",
    version="1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── REQUEST MODELS ───────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class ProposalRequest(BaseModel):
    company_name: str
    company_description: str
    service_type: str = "AI automation"

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

class BusinessNameRequest(BaseModel):
    business_type: str
    style: str = "professional"
    count: int = 5

# ─── ROUTES ───────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AI Automation API",
        "version": "1.0.0",
        "endpoints": [
            "/classify-email",
            "/generate-proposal",
            "/generate-business-names",
            "/summarize",
            "/health"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/classify-email")
def classify_email(request: EmailRequest):
    """Classify an email into category and priority, generate suggested reply."""
    try:
        prompt = f"""Classify this email and respond with ONLY a JSON object:
{{
  "category": "inquiry" or "complaint" or "spam" or "partnership",
  "priority": "high" or "medium" or "low",
  "suggested_reply": "professional 2-3 sentence reply"
}}

From: {request.sender}
Subject: {request.subject}
Body: {request.body}

Respond with ONLY the JSON."""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        import json
        text = response.text.strip()
        # Clean any markdown
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)

        return {
            "success": True,
            "data": result,
            "processed_at": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-proposal")
def generate_proposal(request: ProposalRequest):
    """Generate a personalized AI automation proposal for a business."""
    try:
        prompt = f"""You are an AI automation consultant. Write a professional 
proposal (200-250 words) for this business:

Company: {request.company_name}
Description: {request.company_description}
Service: {request.service_type}

Include:
1. Personalized opening
2. 3 specific AI opportunities
3. Expected ROI
4. Call to action

Write only the proposal text."""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return {
            "success": True,
            "data": {
                "company": request.company_name,
                "proposal": response.text.strip()
            },
            "generated_at": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-business-names")
def generate_business_names(request: BusinessNameRequest):
    """Generate creative business names for Pakistani market."""
    try:
        prompt = f"""Generate {request.count} unique business names for:
Business type: {request.business_type}
Style: {request.style}
Market: Pakistani

For each name provide name, reason, and tagline.
Respond as JSON array:
[{{"name": "...", "reason": "...", "tagline": "..."}}]

Respond with ONLY the JSON array."""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        import json
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        names = json.loads(text)

        return {
            "success": True,
            "data": names,
            "count": len(names)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
def summarize_text(request: TextRequest):
    """Summarize any text in 3 bullet points."""
    try:
        prompt = f"""Summarize this text in exactly 3 clear bullet points.
Respond as JSON:
{{"summary": ["point 1", "point 2", "point 3"]}}

Text: {request.text[:3000]}

Respond with ONLY the JSON."""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        import json
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))