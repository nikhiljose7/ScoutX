from flask import Blueprint, request, jsonify, session

import google.generativeai as genai

# Conversation history storage
conversation_histories = {}
import os
import pandas as pd
from typing import Optional, Dict, Any

import live_api

chatbot_bp = Blueprint('chatbot_bp', __name__)

import google.generativeai.types as types

# Configure the Gemini API (optional; if key not set, model calls will fail later)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def _load_player_data() -> pd.DataFrame:
    """Load the preferred cleaned CSV if available, otherwise fallback to predictions CSV."""
    candidates = [
        os.path.join('moneyball_report_outputs', 'data chatbot.csv'),
        os.path.join('moneyball_report_outputs', 'data chatbot.csv')
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                df = pd.read_csv(p)
                if 'Player' in df.columns:
                    df['Player'] = df['Player'].astype(str)
                return df
        except Exception:
            continue
    return pd.DataFrame()


data = _load_player_data()


# Construct a small dataset context for the model
if not data.empty and 'Undervaluation' in data.columns:
    top_undervalued = data.sort_values(by='Undervaluation', ascending=False).head(5)
    initial_context = (
        "You are a football analytics assistant. For context, here are the top 5 undervalued players from the dataset:\n"
        + top_undervalued.to_string()
    )
else:
    initial_context = "You are a football analytics assistant. Use available player data and public knowledge to answer questions about players, performance and market value."


# Use Gemini model wrapper
model = genai.GenerativeModel('gemini-2.0-flash')


def _find_player_by_name(name: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Return the first matching row as a dict for case-insensitive exact or substring match."""
    if df is None or df.empty or 'Player' not in df.columns:
        return None
    name_norm = str(name).strip().lower()
    players = df['Player'].astype(str).str.strip()
    # exact match first
    mask_exact = players.str.lower() == name_norm
    matches = df[mask_exact]
    if matches.empty:
        mask_contains = players.str.lower().str.contains(name_norm, na=False)
        matches = df[mask_contains]
    if matches.empty:
        return None
    # return first match as dict
    return matches.iloc[0].to_dict()


def serialize_history(history):
    return [{"role": msg.role, "parts": [part.text for part in msg.parts]} for msg in history]


@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Simple chatbot endpoint that prefers API-Football live facts when available."""
    payload = request.json or {}
    message = payload.get('message', '')
    requested_player = payload.get('player')
    use_live = payload.get('use_live', True)

    if not message:
        return jsonify({'reply': 'Please enter a message.'})

    try:
        # start chat with optional session history
        chat_history = session.get('chat_history', [])
        
        # Try to use Simple Direct Search RAG (No APIs, No PyTorch!)
        try:
            from rag_service_simple import get_rag_response
            rag_response = get_rag_response(message)
            
            # Update history (simple append for now)
            chat_history.append({"role": "user", "parts": [message]})
            chat_history.append({"role": "model", "parts": [rag_response]})
            session['chat_history'] = chat_history
            
            return jsonify({'reply': rag_response})
        except Exception as e:
            import traceback
            with open('rag_error.log', 'w') as f:
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
            print(f"RAG Error: {e}")
            return jsonify({'reply': "I'm sorry, I couldn't retrieve the information from my database at this time."})
    except Exception as e:
        return jsonify({'reply': f'An error occurred: {e}'})



@chatbot_bp.route('/api/chatbot/reset', methods=['POST'])
def reset_chatbot():
    session.pop('chat_history', None)
    return jsonify({'status': 'Chat history reset'})


@chatbot_bp.route('/api/sofascore/tv-countries', methods=['GET'])
def sofascore_tv_countries():
    """Proxy endpoint to fetch available TV countries/channels for a SofaScore match.

    Query parameter: matchId (int)
    Returns SofaScore RapidAPI JSON or an error message. The RapidAPI key must be set
    in the environment variable `SOFASCORE_RAPIDAPI_KEY` for this to work.
    """
    match_id = request.args.get('matchId') or request.args.get('match_id')
    if not match_id:
        return jsonify({'success': False, 'error': 'Provide matchId query parameter'}), 400
    try:
        mid = int(match_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'matchId must be an integer'}), 400

    try:
        j = live_api.get_match_tv_countries(mid)
        if j is None:
            return jsonify({'success': False, 'error': 'No data or API key not configured for SofaScore RapidAPI.'}), 500
        return jsonify({'success': True, 'data': j})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

