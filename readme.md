---------

# Chatbot with PyTorch

## Overview

This project demonstrates a simple chatbot built using PyTorch, Natural Language Processing (NLP), and a feedforward neural network. The chatbot is trained to recognize intents from user input and provide appropriate responses. The model is based on a bag-of-words approach and uses stemming to process the input data.

## Features

-   **Neural Network-based Model**: Utilizes a feedforward neural network with ReLU activation to classify user intents.
-   **Natural Language Processing**: Tokenizes and stems user input to generate a bag-of-words representation for prediction.
-   **Training and Inference**: The chatbot is trained on custom intents and responds based on its predictions.

## Files Overview

1.  **`model.py`**: Defines the neural network model with three layers.
2.  **`nltk_utils.py`**: Contains utility functions for tokenizing and stemming input data, as well as converting it to bag-of-words format.
3.  **`train.py`**: Loads training data, trains the model, and saves the trained state.
4.  **`chat.py`**: The interactive chatbot interface that loads the trained model and generates responses to user input.
5.  **`intents.json`**: A JSON file containing predefined intents, patterns, and responses.

## How to Run

### 1. Install Dependencies

```bash
pip install torch nltk numpy

```

### 2. Train the Model

Run the `train.py` script to train the chatbot:

```bash
python train.py

```

This will process the training data and save the trained model to `data.pth`.

### 3. Chat with the Bot

Once the model is trained, run `chat.py` to start interacting with the bot:

```bash
python chat.py

```

## Example Interaction

```
Let's chat! (type 'quit' to exit)
You: Hi
Bot: Hello! How can I assist you today?

You: Can you tell me a joke?
Bot: Why don't skeletons fight each other? They don't have the guts!

```

## Customization

-   **Add More Intents**: Modify the `intents.json` file to add more patterns and responses.
-   **Train on New Data**: You can retrain the model on new intents by modifying `train.py`.

## License

This project is open-source and available under the MIT License.

----------

