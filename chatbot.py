from flask import Blueprint, request, jsonify, session
import google.generativeai as genai
import os
import pandas as pd

chatbot_bp = Blueprint('chatbot_bp', __name__)

# Configure the Gemini API
genai.configure(api_key=os.getenv("API_KEY_GEMINI"))

# Load the player data
data = pd.read_csv('moneyball_report_outputs/all_predictions_with_undervaluation.csv')

# Pre-computation for context
top_undervalued_players = data.sort_values(by='Undervaluation', ascending=False).head(5)
initial_context = f"You are a football analytics assistant. Here are the top 5 most undervalued players right now:\n{top_undervalued_players.to_string()}"

# Create the generative model with the Gemini 2.0 Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot():
    """API endpoint for the chatbot"""
    message = request.json.get('message', '')
    if not message:
        return jsonify({'reply': 'Please enter a message.'})

    try:
        # Get chat history from the user's session or start a new one
        chat_history = session.get('chat_history', [])
        
        # Start a chat session with the existing history
        chat = model.start_chat(history=chat_history)
        
        # Send the new message to the model
        if not chat_history:
            # If it's the first message, send the initial context as well
            response = chat.send_message(f"{initial_context}\n\nUser question: {message}")
        else:
            response = chat.send_message(message)

        # Update the chat history in the session
        session['chat_history'] = chat.history
        
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'reply': f'An error occurred: {e}'})

@chatbot_bp.route('/api/chatbot/reset', methods=['POST'])
def reset_chatbot():
    """API endpoint to reset the chat history"""
    session.pop('chat_history', None)
    return jsonify({'status': 'Chat history reset'})