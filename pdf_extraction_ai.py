# ✅ Install dependencies (Run this first in Google Colab)
!pip install pymupdf openai spacy
!python -m spacy download en_core_web_sm

# ✅ Import required libraries
import fitz  # PyMuPDF for PDF extraction
import spacy  # NLP for entity recognition
import re
import json
from collections import defaultdict
import openai  # OpenAI API for AI summarization

# ✅ Load NLP model
nlp = spacy.load("en_core_web_sm")

# ✅ Upload your PDF manually in Colab (Run this cell, select the PDF)
from google.colab import files
uploaded = files.upload()

# ✅ Get the filename from uploaded file
pdf_path = list(uploaded.keys())[0]

# ✅ Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# ✅ Clean extracted text
def clean_text(text):
    """Cleans extracted text by removing unwanted spaces and characters."""
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    text = re.sub(r'[^a-zA-Z0-9.,$%\-\s]', '', text)  # Remove special characters
    return text.strip()

# ✅ Extract key financial entities using NLP
def extract_financial_entities(text):
    """Extracts key financial terms using Named Entity Recognition (NER)."""
    doc = nlp(text)
    financial_data = defaultdict(set)
    
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERCENT", "ORG", "DATE", "GPE"]:
            financial_data[ent.label_].add(ent.text)
    
    return {key: list(value) for key, value in financial_data.items()}

# ✅ OpenAI API Key (Replace with your actual key)
OPENAI_API_KEY = "your-api-key-here"
openai.api_key = OPENAI_API_KEY

# ✅ AI-based summarization
def summarize_text(text):
    """Uses GPT-4 to summarize key financial insights for investors."""
    prompt = (
        "Extract key financial insights from the following earnings call transcript. "
        "Summarize details on future growth prospects, key business changes, "
        "triggers affecting growth, and material information for next year's earnings.\n\n"
        f"Transcript:\n{text[:4000]}\n\nSummary:"  # Limiting input size to avoid token limit issues
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a financial analyst."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error in GPT-4 summarization: {str(e)}"

# ✅ Run the pipeline
extracted_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(extracted_text)
financial_entities = extract_financial_entities(cleaned_text)
ai_summary = summarize_text(cleaned_text)

# ✅ Generate Markdown report
def generate_markdown(summary, financial_entities):
    """Generates a professional Markdown report from extracted insights."""
    md_output = f"""# Investor Insights Report

## Summary  
{summary}  

## Financial Data  
"""
    for key, values in financial_entities.items():
        md_output += f"**{key}:** {', '.join(values)}\n\n"
    
    return md_output

markdown_report = generate_markdown(ai_summary, financial_entities)

# ✅ Save Markdown report to a file
with open("investor_report.md", "w") as file:
    file.write(markdown_report)

# ✅ Print the structured output
structured_output = {
    "summary": ai_summary,
    "financial_entities": financial_entities,
    "key_findings": {
        "growth_prospects": "Extracted from summary",
        "business_changes": "Extracted from summary",
        "growth_triggers": "Extracted from summary",
        "material_effects": "Extracted from summary"
    }
}

print(json.dumps(structured_output, indent=4))

# ✅ Download the report from Colab
from google.colab import files
files.download("investor_report.md")
