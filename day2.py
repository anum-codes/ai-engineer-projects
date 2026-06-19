from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#creating a function to generate business names
def generate_business_names(business_type, style):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            {"role":"user",
             "parts":[
                 {
                    "text":f"""You are a creative brand naming expert specializing in Pakistani Businesses. You understand Urdu, English and local market deeply.
                     
                    When given a business type and style, generate five unique, memorable business names.
                     For each name provide:
                     - The name itself
                     - why it works
                     - A one-line tagline

                    Business Type: {business_type}
                    Style: {style}
                    Keep names catchy, proffessional and relevant to pakistani market.
                     """
                     }
             ]
             }
        ]
    )

    return response.text

# Testing it with 3 different combinations
print("="*50)
print("TECH STARTUP - MODERN STYLE")
print("=" * 50)
print(generate_business_names("AI automation agency", "modern and tech-savvy"))

print("\n" + "=" * 50)
print("FOOD BUSINESS - TRADITIONAL STYLE")
print("=" * 50)
print(generate_business_names("online food delivery", "traditional Pakistani"))

print("\n" + "=" * 50)
print("FREELANCE SERVICES - PROFESSIONAL STYLE")
print("=" * 50)
print(generate_business_names("WordPress and AI services", "professional and trustworthy"))
