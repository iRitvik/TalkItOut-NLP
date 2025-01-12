import torch
import torch.nn as nn
import random
import json
from flask import Flask, render_template, request, jsonify

# Define the Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Helper functions for tokenizing and creating bag of words
def tokenize(sentence):
    return sentence.split()

def bag_of_words(tokenized_sentence, all_words):
    bag = [0] * len(all_words)
    for w in tokenized_sentence:
        for idx, word in enumerate(all_words):
            if word == w:
                bag[idx] = 1
    return bag

# Load the model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('functions/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "functions/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X).float().to(device)

    output = model(X.unsqueeze(0))
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Check your mail, I have sent you some fun activities to do!"

# Flask App
app = Flask(__name__)

@app.route('/')
def index_get():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)