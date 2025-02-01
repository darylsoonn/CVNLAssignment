import torch
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Define the same model architecture as q2_intent.py
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = torch.load('intent_model_full.pth', map_location=torch.device('cpu'))
model.eval()

# Load tokenizer (vocabulary)
with open('tokenizer.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Custom tokenizer function
def custom_tokenizer(sentence):
    import re
    sentence = sentence.lower()
    tokens = re.findall(r"\b\w+\b", sentence)
    return tokens

# Preprocess input text
def preprocess_text(text, max_length=30):
    tokens = [vocab[word] if word in vocab else vocab['<UNK>'] for word in custom_tokenizer(text)]
    tokens = tokens[:max_length]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sentence = data.get('sentence')

        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400

        input_tensor = preprocess_text(sentence)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            label = label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({'predicted_label': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
