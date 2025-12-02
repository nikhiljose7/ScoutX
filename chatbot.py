from flask import Blueprint, request, jsonify, session
import google.generativeai as genai
import os
import pandas as pd
from typing import Optional, Dict, Any

import live_api

chatbot_bp = Blueprint('chatbot_bp', __name__)

import google.generativeai.types as types

# Monkeypatch for GenerationConfig.MediaResolution error
try:
    if not hasattr(types.GenerationConfig, 'MediaResolution'):
        class MediaResolution:
            UNSPECIFIED = 0
            LOW = 1
            MEDIUM = 2
            HIGH = 3
        types.GenerationConfig.MediaResolution = MediaResolution
except Exception:
    pass

# Configure the Gemini API (optional; if key not set, model calls will fail later)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def _load_player_data() -> pd.DataFrame:
    """Load the preferred cleaned CSV if available, otherwise fallback to predictions CSV."""
    candidates = [
        os.path.join('moneyball_report_outputs', 'players_data_cleaned_with_market_values_with_market_values1.csv'),
        os.path.join('moneyball_report_outputs', 'all_predictions_with_undervaluation.csv')
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
        chat = model.start_chat(history=chat_history)

        # Detect player from explicit param or message text
        player_row = None
        if requested_player:
            player_row = _find_player_by_name(requested_player, data)
        if player_row is None and not data.empty:
            low = message.lower()
            for candidate in data['Player'].astype(str).unique():
                if candidate and candidate.lower() in low:
                    player_row = _find_player_by_name(candidate, data)
                    if player_row:
                        break

        # Use RAG service for better answers using the vector database
        try:
            from rag_service import rag_service
        except ImportError:
            # Fallback if rag_service fails to import
            rag_service = None
        
        # Combine user message with any context we found (like live data)
        # Add live facts if available
        live_info = None
        recent_matches = None
        
        if use_live:
            try:
                # 1. Player search
                pname = requested_player or (player_row.get('Player') if player_row else None)
                if pname:
                    live_info = live_api.get_player_live_summary(pname)
                
                # 2. Team/Match search
                low_msg = message.lower()
                if any(k in low_msg for k in ['match', 'game', 'score', 'result', 'play']):
                    import re
                    potential_teams = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', message)
                    
                    ignore = {'How', 'What', 'When', 'Where', 'Who', 'Why', 'The', 'A', 'An', 'Is', 'Are', 'Was', 'Were', 'Did', 'Does', 'Do', 'Can', 'Could', 'Will', 'Would', 'Should', 'May', 'Might', 'Must', 'Have', 'Has', 'Had', 'Tell', 'Me', 'About', 'Show', 'Give', 'List', 'Find', 'Get', 'Recent', 'Last', 'Next', 'Live', 'Match', 'Game', 'Score', 'Result', 'Played', 'Playing', 'Vs', 'Versus', 'Against'}
                    candidates = [t for t in potential_teams if t not in ignore and len(t) > 3]
                    
                    if candidates:
                        team_name = candidates[0]
                        recent_matches = live_api.get_team_recent_matches(team_name)
                    
                    if 'live' in low_msg:
                        live_summary = live_api.get_live_matches_summary()
                        if live_summary:
                            if recent_matches:
                                recent_matches += "\n\n" + live_summary
                            else:
                                recent_matches = live_summary

            except Exception as e:
                print(f"Live API error: {e}")
                live_info = None

        # Use RAG service for better answers using the vector database
        # from rag_service import rag_service (already imported above)
        
        # Combine user message with any context we found (like live data)
        rag_query = message
        if live_info:
            rag_query += f"\n\nLive Info Context: {live_info}"
        if recent_matches:
            rag_query += f"\n\nRecent Matches Context: {recent_matches}"
        if player_row:
             rag_query += f"\n\nPlayer Data Context: {player_row}"

        if rag_service:
            reply = rag_service.get_answer(rag_query)
        else:
            reply = "RAG service is unavailable."
        
        # Update chat history manually since we are not using chat.send_message directly anymore
        # or we can just append to session if we want to maintain history for RAG too, 
        # but RAG usually handles single turn or we need to pass history to RAG.
        # For now, let's just return the reply.
        
        # If we want to keep using the chat session for history, we can feed the RAG answer back to it,
        # but RAG is a replacement for the generation part.
        
        # Let's just return the RAG reply.
        return jsonify({'reply': reply})
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
