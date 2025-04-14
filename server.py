from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load model
model = BertForSequenceClassification.from_pretrained("cognitive_model")
tokenizer = BertTokenizer.from_pretrained("cognitive_model")
model.eval()

class Transcript(BaseModel):
    text: str

@app.post("/analyze-text")
def analyze_text(data: Transcript):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        dementia_prob = float(probs[0][1])
    return {"dementia_risk": round(dementia_prob * 100, 2)}
