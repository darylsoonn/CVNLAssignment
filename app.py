from flask import Flask, request, jsonify
import torch
import pickle
import nltk
from nltk.tokenize import word_tokenize
import torch.nn as nn
from flask_cors import CORS
import torch.nn.functional as F  # Importing softmax

# Download the required NLTK data
nltk.download('punkt')

# Define the RNN model class
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

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        outputs, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])
    

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
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allowing CORS for /predict route

# Load the trained model
model = torch.load('RNN_Q.pth')
model.eval()  # Set the model to evaluation mode

# Load the trained intent model
intent_model = IntentRNNModel(vocab_size=10000, embed_dim=100, hidden_dim=128, output_dim=10)  # Adjust these parameters as needed
intent_model.load_state_dict(torch.load('intent_model_full.pth'))
intent_model.eval()

# Load the tokenizer from the pickle file
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']  # Your actual labels

with open('intent_tokenizer.pkl', 'rb') as f:
    intent_tokenizer = pickle.load(f)



def preprocess_text(text, max_length=30):
    # Tokenize the text and map tokens to their IDs using the tokenizer
    tokens = word_tokenize(text.lower())
    print(f"Tokens: {tokens}")  # Debugging line
    token_ids = [tokenizer.get(word, tokenizer.get('<UNK>')) for word in tokens]  # Handle unknown tokens
    token_ids = token_ids[:max_length]  # Limit to max_length tokens
    print(f"Token IDs: {token_ids}")  # Debugging line
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from user
        data = request.get_json()
        sentence = data.get('sentence')
        print(f"Received sentence: {sentence}")  # Log the input sentence

        # Check if sentence is provided
        if not sentence:
            print("No sentence provided")
            return jsonify({'error': 'No sentence provided'}), 400

        # Preprocess text and make prediction
        input_tensor = preprocess_text(sentence)
        print(f"Processed input tensor: {input_tensor}")  # Log the tensor after preprocessing

        with torch.no_grad():
            output = model(input_tensor)
            print(f"Model output (raw logits): {output}")  # Log the raw output
            output = F.softmax(output, dim=1)  # Apply softmax to get probabilities
            print(f"Softmax probabilities: {output}")  # Log probabilities
            predicted_label = torch.argmax(output, dim=1).item()

            # Directly map the predicted index to the emotion
            label = emotion_labels[predicted_label]
            print(f"Predicted label: {label}")  # Log the final label

        return jsonify({'predicted_label': label})

    except Exception as e:
        print(f"Error: {e}")  # Log any exception
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
