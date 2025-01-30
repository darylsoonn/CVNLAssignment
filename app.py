from flask import Flask, request, jsonify
import torch
import pickle
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn as nn
from flask_cors import CORS  

class EmotionRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(EmotionRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
        
class IntentRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(IntentRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
    
# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained emotion model
emotion_model = EmotionRNNModel(vocab_size=10000, embed_dim=100, hidden_dim=128, output_dim=5)  # Adjust these parameters as needed
emotion_model.load_state_dict(torch.load('RNN_Q.pth'))
emotion_model.eval()

# Load the trained intent model
intent_model = IntentRNNModel(vocab_size=10000, embed_dim=100, hidden_dim=128, output_dim=10)  # Adjust these parameters as needed
intent_model.load_state_dict(torch.load('intent_model_full.pth'))
intent_model.eval()

# Load the tokenizers and label encoders
with open('emotion_tokenizer.pkl', 'rb') as f:
    emotion_tokenizer = pickle.load(f)

with open('intent_tokenizer.pkl', 'rb') as f:
    intent_tokenizer = pickle.load(f)

# Emotion label encoder
emotion_label_encoder = LabelEncoder()
emotion_label_encoder.classes_ = np.load('emotion_label_encoder_classes.npy', allow_pickle=True)

# Intent label encoder
intent_label_encoder = LabelEncoder()
intent_label_encoder.classes_ = np.load('intent_label_encoder_classes.npy', allow_pickle=True)


# Tokenizer function for emotion model
def preprocess_emotion_text(text, max_length=30):
    tokens = [emotion_tokenizer[word] if word in emotion_tokenizer else emotion_tokenizer['<UNK>'] for word in word_tokenize(text.lower())]
    tokens = tokens[:max_length]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension


# Tokenizer function for intent model
def preprocess_intent_text(text, max_length=30):
    tokens = [intent_tokenizer[word] if word in intent_tokenizer else intent_tokenizer['<UNK>'] for word in word_tokenize(text.lower())]
    tokens = tokens[:max_length]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension


# Emotion prediction endpoint
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        # Get input from user
        data = request.get_json()
        sentence = data.get('sentence')

        # Check if sentence is provided
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400

        # Preprocess text and make prediction
        input_tensor = preprocess_emotion_text(sentence)
        with torch.no_grad():
            output = emotion_model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            label = emotion_label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({'predicted_label': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Intent prediction endpoint
@app.route('/predict_intent', methods=['POST'])
def predict_intent():
    try:
        # Get input from user
        data = request.get_json()
        sentence = data.get('sentence')

        # Check if sentence is provided
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400

        # Preprocess text and make prediction
        input_tensor = preprocess_intent_text(sentence)
        with torch.no_grad():
            output = intent_model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            label = intent_label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({'predicted_label': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
