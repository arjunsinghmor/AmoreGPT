from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and tokenizer (DialoGPT-medium in this case)
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")

# Global variable to store chat history
chat_history_ids = None

@app.route('/')
def index():
    # Serve the frontend HTML page
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids

    # Get the user input from the frontend
    user_input = request.json.get('user_input')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # Encode the user input and add the end-of-sequence token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # If there is previous chat history, concatenate it with the new user input
    bot_input_ids = new_user_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    
    # Generate a response from the model (with some history context)
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the model's response and skip the special tokens
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Return the chatbot's response to the frontend
    return jsonify({'response': bot_output})

if __name__ == '__main__':
    # Run the Flask app on all available IPs (0.0.0.0) for Docker container access
    app.run(host='0.0.0.0', port=5000)