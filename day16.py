from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

@app.get("/")
def home():
    return {"message": "AI Automation API is running"}

@app.post("/classify-email")
def classify_email(request: EmailRequest):
    try:
        # Build the prompt using data from the request
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

        # Call Gemini — same as Day 1
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        # Clean and parse the response
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class SummarizeRequest(BaseModel):
    text: str

@app.post('/summarize')
def summarize(request: SummarizeRequest):
    try:
        prompt = f"""Summarize this text in exactly 3 bullet points and respond with only this JSON format:
        {{"summary": ["point 1", "point 2", "point 3"]}}

        text: {request.text}

        Respond with ONLY the JSON."""
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        ftext = response.text.strip()
        ftext = ftext.replace("```json", "").replace("```", "").strip()
        result = json.loads(ftext)

        return {
            "success": True,
            "summary": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class BusinessNameRequest(BaseModel):
    business_type: str
    style: str

@app.post('/generate-business-names')
def generate_business_names(request: BusinessNameRequest):
    try:
        prompt = f"""You are a creative brand naming expert. Generate 3 unique business names.

        Business type: {request.business_type}
        Style: {request.style}

        Respond with ONLY this JSON array, nothing else:
        [
            {{"name": "name here", "tagline": "tagline here", "reason": "why it works here"}},
            {{"name": "name here", "tagline": "tagline here", "reason": "why it works here"}},
            {{"name": "name here", "tagline": "tagline here", "reason": "why it works here"}}
        ]"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
            )
    
        ftext = response.text.strip()
        ftext = ftext.replace("```json", "").replace("```", "").strip()
        result = json.loads(ftext)

        return {"success": True, "Business_Names": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))