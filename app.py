from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load tokenizer dan model dari Hugging Face Hub
model_path = "BangBam/indobert_accuracy_10000ds"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        if prediction == 0:
            label = "Negatif"
        elif prediction == 1:
            label = "Netral"
        else:
            label = "Positif"

        return render_template('index.html', prediction=label, text=text)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
