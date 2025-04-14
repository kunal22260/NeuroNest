from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, BertTokenizer, BertModel
import torch
import soundfile as sf
import random

# Load models once
wav_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wav_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def transcribe_audio(audio_path):
    speech, rate = sf.read(audio_path)
    input_values = wav_tokenizer(speech, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = wav_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return wav_tokenizer.batch_decode(predicted_ids)[0]

def analyze_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return simulate_risk_score(text)

def simulate_risk_score(text):
    word_count = len(text.split())
    if word_count < 20:
        return round(random.uniform(0.7, 0.95), 2)
    else:
        return round(random.uniform(0.1, 0.5), 2)
