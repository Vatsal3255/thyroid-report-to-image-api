import os
import json
import sys
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .config_store import load_api_key

# Ensure environment variables from a local .env file are loaded before use.
load_dotenv()

# ---------------------------------------------------------------------------------
YOUR_API_KEY = os.getenv("GEMINI_API_KEY")
# ---------------------------------------------------------------------------------

class Nodule(BaseModel):
    lobe: str = Field(description="The thyroid lobe location, e.g., 'Right Lobe', 'Left Lobe', 'Isthmus'.")
    region: str = Field(description="The region within the lobe, e.g., 'upper', 'lower', 'middle', 'lower pole'. Should be 'N/A' if not specified.")
    size_mm: str = Field(description="Largest dimension size of the nodule in mm, e.g., '12x8x10 mm' or '1cm x 1cm'.")
    echogenicity: str = Field(description="Description of echogenicity, e.g., 'hypoechoic', 'isoechoic', 'anechoic'.")
    composition: str = Field(description="Composition description, e.g., 'solid', 'predominantly cystic', 'cyst-solid mixed nodule'.")
    margins: str = Field(description="Margin description, e.g., 'smooth', 'lobulated', 'ill-defined margin'.")
    calcifications: str = Field(description="Presence and type of calcifications, e.g., 'microcalcifications', 'none'. Infer 'none' if not mentioned.")
    ti_rads_score: Optional[str] = Field(description="The TI-RADS score, if explicitly mentioned, e.g., 'TI-RADS 4', 'TIRADS 1'.")

class ThyroidUltrasoundReport(BaseModel):
    """Overall schema for the thyroid ultrasound report summary."""
    ultrasound_date: str = Field(description="Date of the ultrasound. Prefer ISO format YYYY-MM-DD. Use 'Not mentioned' if unavailable.")
    gland_size: str = Field(description="General statement on gland size or dimensions.")
    overall_echotexture: str = Field(description="General description of the background thyroid echotexture, e.g., 'homogeneous', 'heterogeneous', 'uneven'.")
    nodules: List[Nodule] = Field(description="A list of all distinct nodules identified in the report.")
    lymph_nodes: str = Field(description="Description of cervical lymph nodes. If not mentioned, use 'Not mentioned'.")
    impression_or_conclusion: str = Field(description="The final clinical impression or summary conclusion of the report.")


def extract_thyroid_info(report_text: str, api_key: str) -> dict:
    if not api_key or api_key == "YOUR_API_KEY":
        print("\nFATAL ERROR: Please replace the placeholder API key in the script with your actual key.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return {}
    
    # Construct the prompt
    system_instruction = (
        "You are an expert medical data extraction tool specializing in radiology reports. "
        "Your task is to accurately parse the provided thyroid ultrasound report and extract "
        "all specified clinical details into the required JSON schema format. Extract the exact ultrasound date as written "
        "in the report (no adjustments or day/month swapping) and normalize it to YYYY-MM-DD; if absent return 'Not mentioned'. "
        "Ensure the nodule's **LOBE** (Right, Left, Isthmus) and **REGION** (upper, lower, middle, etc.) "
        "are extracted into separate fields. Only output the requested JSON object."
    )
    
    prompt = f"""
    Analyze the following thyroid ultrasound report and extract the details into the structured format.

    Convert the Region to upper, upper middle, middle, lower middle, lower from the word of the report (e.g. superior -> upper).
    Extract the ultrasound date exactly as written in the report with no day/month swapping or timezone shifts, and output it as ISO YYYY-MM-DD.
    Examples:
      "29 Nov, 2025" -> "2025-11-29"
      "12/03/2024" (dd/MM/yyyy) -> "2024-03-12"
      If no date, return "Not mentioned".
    
    THYROID ULTRASOUND REPORT:
    ---
    {report_text}
    ---
    """
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=ThyroidUltrasoundReport,
        temperature=0.0
    )
    
    print("Sending request to Gemini API...")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        
        extracted_data_model = response.parsed
        return extracted_data_model.model_dump()
        
    except Exception as e:
        print(f"An error occurred during API call or parsing: {e}")
        return {}

def extract_info(sample_report):
    if sample_report == "":
        print("Report not found...")
        return None
    api_key = YOUR_API_KEY or load_api_key()
    if not api_key:
        print("API Key Not Found.")
        return None
    extracted_info = extract_thyroid_info(sample_report, api_key)
    if extracted_info:
        return extracted_info
    return None
