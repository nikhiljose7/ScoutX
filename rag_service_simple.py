"""
Simple Direct Search RAG - Updated for data chatbot.csv
Searches directly in CSV data using fuzzy matching
"""
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Global data cache
chatbot_df = None

def load_data():
    """Load chatbot CSV file into memory"""
    global chatbot_df
    
    if chatbot_df is None:
        print("[RAG] Loading chatbot data...")
        try:
            chatbot_df = pd.read_csv('moneyball_report_outputs/data chatbot.csv')
            print(f"[RAG] Loaded {len(chatbot_df)} players from data chatbot.csv")
            print(f"[RAG] Columns: {list(chatbot_df.columns)}")
        except Exception as e:
            print(f"[RAG] Error loading data: {e}")

def fuzzy_match_score(str1, str2):
    """Calculate fuzzy match score between two strings"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def search_player(query):
    """Search for a player by name using fuzzy matching"""
    global chatbot_df
    
    load_data()
    
    if chatbot_df is None:
        return []
    
    query_lower = query.lower()
    print(f"[RAG] Searching for: '{query}'")
    
    # Try exact match first
    exact_matches = chatbot_df[chatbot_df['Player'].str.lower() == query_lower]
    if not exact_matches.empty:
        print(f"[RAG] Found exact match: {exact_matches.iloc[0]['Player']}")
        return exact_matches.to_dict('records')
    
    # Try substring match
    substring_matches = chatbot_df[chatbot_df['Player'].str.contains(query, case=False, na=False)]
    if not substring_matches.empty:
        print(f"[RAG] Found {len(substring_matches)} substring matches")
        return substring_matches.to_dict('records')[:5]
    
    # Fuzzy match
    chatbot_df['match_score'] = chatbot_df['Player'].apply(
        lambda x: fuzzy_match_score(str(x), query) if pd.notna(x) else 0
    )
    
    fuzzy_matches = chatbot_df[chatbot_df['match_score'] > 0.5].sort_values('match_score', ascending=False).head(5)
    
    if not fuzzy_matches.empty:
        print(f"[RAG] Found {len(fuzzy_matches)} fuzzy matches")
        for idx, row in fuzzy_matches.iterrows():
            print(f"  - {row['Player']} (score: {row['match_score']:.2f})")
        return fuzzy_matches.to_dict('records')
    
    print("[RAG] No matches found")
    return []

def search_general(query):
    """Search for general queries"""
    global chatbot_df
    
    load_data()
    
    if chatbot_df is None:
        return []
    
    query_lower = query.lower()
    keywords = query_lower.split()
    
    print(f"[RAG] General search for: '{query}'")
    
    # Search for undervalued players
    if 'undervalued' in query_lower or 'undervalue' in query_lower:
        if chatbot_df is not None and 'Undervaluation' in chatbot_df.columns:
            # Filter out non-numeric values
            chatbot_df['Undervaluation_numeric'] = pd.to_numeric(chatbot_df['Undervaluation'], errors='coerce')
            top_undervalued = chatbot_df[chatbot_df['Undervaluation_numeric'].notna()].sort_values('Undervaluation_numeric', ascending=False).head(10)
            print(f"[RAG] Found top 10 undervalued players")
            return top_undervalued.to_dict('records')
    
    # Search by position
    for keyword in keywords:
        if keyword.upper() in ['FW', 'MF', 'DF', 'GK', 'FORWARD', 'MIDFIELDER', 'DEFENDER', 'GOALKEEPER']:
            if 'Position' in chatbot_df.columns:
                pos_matches = chatbot_df[chatbot_df['Position'].str.contains(keyword, case=False, na=False)].head(10)
                if not pos_matches.empty:
                    print(f"[RAG] Found {len(pos_matches)} players by position")
                    return pos_matches.to_dict('records')
    
    # Search by team
    if 'Team' in chatbot_df.columns:
        for keyword in keywords:
            team_matches = chatbot_df[chatbot_df['Team'].str.contains(keyword, case=False, na=False)].head(10)
            if not team_matches.empty:
                print(f"[RAG] Found {len(team_matches)} players by team")
                return team_matches.to_dict('records')
    
    return []

def safe_float_format(value, prefix="€", suffix="M"):
    """Safely format a value that might be string or number"""
    try:
        if pd.isna(value):
            return None
        # Try to convert to float
        num_val = float(value)
        return f"{prefix}{num_val:.2f}{suffix}"
    except (ValueError, TypeError):
        # If it's "not available" or other string, just return as is
        return f"{prefix}{value}{suffix}"

def format_player_context(players_data):
    """Format player data into readable context"""
    if not players_data:
        return ""
    
    context_parts = []
    for player in players_data:
        parts = [f"Player: {player.get('Player', 'Unknown')}"]
        
        if 'Team' in player and pd.notna(player['Team']):
            parts.append(f"Team: {player['Team']}")
        if 'league' in player and pd.notna(player['league']):
            parts.append(f"League: {player['league']}")
        if 'Age' in player and pd.notna(player['Age']):
            parts.append(f"Age: {player['Age']}")
        if 'Age_Description' in player and pd.notna(player['Age_Description']):
            parts.append(f"Age Category: {player['Age_Description']}")
        if 'Position' in player and pd.notna(player['Position']):
            parts.append(f"Position: {player['Position']}")
        if 'Nation' in player and pd.notna(player['Nation']):
            parts.append(f"Nationality: {player['Nation']}")
        if 'Born' in player and pd.notna(player['Born']):
            try:
                parts.append(f"Birth Year: {int(float(player['Born']))}")
            except:
                pass
        
        # Safely format market values
        if 'Market_Value_Million_EUR' in player:
            formatted = safe_float_format(player['Market_Value_Million_EUR'])
            if formatted:
                parts.append(f"Market Value: {formatted}")
        
        if 'Predicted_Value' in player:
            formatted = safe_float_format(player['Predicted_Value'])
            if formatted:
                parts.append(f"Predicted Value: {formatted}")
        
        if 'Undervaluation' in player:
            formatted = safe_float_format(player['Undervaluation'])
            if formatted:
                parts.append(f"Undervaluation: {formatted}")
        
        # Add key stats if available
        stats = []
        if 'Goals' in player and pd.notna(player['Goals']):
            try:
                stats.append(f"Goals: {int(float(player['Goals']))}")
            except:
                pass
        if 'Assists' in player and pd.notna(player['Assists']):
            try:
                stats.append(f"Assists: {int(float(player['Assists']))}")
            except:
                pass
        if 'Matches Played' in player and pd.notna(player['Matches Played']):
            try:
                stats.append(f"Matches: {int(float(player['Matches Played']))}")
            except:
                pass
        
        if stats:
            parts.append(f"Stats: {', '.join(stats)}")
        
        context_parts.append(" | ".join(parts))
    
    return "\n\n".join(context_parts)

def get_rag_response(query, history=None):
    """Get RAG response with enhanced prompts and conversation history"""
    try:
        print(f"\n[RAG] Processing query: '{query}'")
        
        # Check for greetings and common questions
        query_lower = query.lower().strip()
        greetings = ['hi', 'hai', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'hola', 'greetings']
        help_words = ['help', 'what can you do', 'how do you work', 'what do you do', 'commands']
        
        # Greeting response
        if any(greeting in query_lower for greeting in greetings) and len(query_lower) < 20:
            return """Hey there! 👋 Welcome to ScoutX Football Analytics!

I'm here to help you discover amazing football players and insights! Here's what I can do:

⚽ **Player Analysis**
- Ask about any player: "Tell me about Mason Greenwood"
- Get detailed stats, performance metrics, and market value insights

🔍 **Player Comparisons**
- Compare multiple players: "Compare Rashford and Greenwood"
- See side-by-side stats and recommendations

💰 **Find Hidden Gems**
- Ask: "Show me undervalued forwards"
- Discover smart signing opportunities

🎯 **Team Analysis**
- "Best players from Manchester United"
- "Top scorers in Premier League"

**Just ask me anything about football players, and I'll help you out! 😊**

Try asking: "Who is Erling Haaland?" or "Best young midfielders" 🚀"""
        
        # Help response
        if any(word in query_lower for word in help_words):
            return """**Here's how I can help you! 🤝**

📊 **What I Know:**
- Detailed stats for 2,699+ players
- Performance metrics (goals, assists, matches, etc.)
- Market value analysis and predictions
- Undervaluation insights

💡 **How to Ask:**
- **Single player**: "Tell me about [player name]"
- **Comparison**: "Compare [player 1] and [player 2]"
- **Position search**: "Best strikers" or "Top defenders"
- **Team search**: "Players from Real Madrid"
- **Value search**: "Undervalued midfielders"

**Examples to try:**
- "Who is Kylian Mbappé?"
- "Compare Haaland and Kane"
- "Best young wingers"
- "Undervalued players in La Liga"

Ready to discover some amazing talent? Just ask away! ⚽✨"""
        
        results = search_player(query)
        
        if not results:
            results = search_general(query)
        
        if not results:
            return "I couldn't find any relevant information. Try asking about specific players, teams, or use keywords like 'undervalued players'."
        
        context = format_player_context(results)
        print(f"[RAG] Found {len(results)} results")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        is_single_player = len(results) == 1
        
        if is_single_player:
            # Add conversation history if available
            history_context = ""
            if history and len(history) > 0:
                history_context = "\nPrevious conversation context:\n"
                for h in history[-3:]:  # Last 3 exchanges
                    history_context += f"User: {h['user']}\nAssistant: {h['assistant'][:150]}...\n"
            
            prompt = f"""You are a friendly football analytics expert who loves helping people discover amazing players! 😊 Create an engaging and informative player report.
{history_context}
PLAYER DATA:
{context}

USER QUESTION: {query}

Generate a detailed, professional player report:

📊 **PLAYER PROFILE**
- Name, age, position, team, league, nationality

⚽ **PERFORMANCE ANALYSIS**
- Age category and career stage
- Key statistics (goals, assists, matches)
- Strengths and standout attributes

💰 **MARKET VALUE ANALYSIS**
- Current market value
- Predicted value (from our model)
- Undervaluation status and investment recommendation

📈 **KEY INSIGHTS**
- Notable characteristics
- Market opportunity assessment
- Suitability for different team strategies

Use emojis and clear formatting. Be friendly, enthusiastic, and conversational while staying informative! 🚀"""

        else:
            # Add conversation history if available
            history_context = ""
            if history and len(history) > 0:
                history_context = "\nPrevious conversation context:\n"
                for h in history[-3:]:  # Last 3 exchanges
                    history_context += f"User: {h['user']}\nAssistant: {h['assistant'][:150]}...\n"
            
            prompt = f"""You are a friendly football analytics expert who makes player comparisons fun and insightful! 😊
{history_context}
PLAYERS DATA:
{context}

USER QUESTION: {query}

Provide a clear, organized response:
1. Present players in a ranked format if applicable
2. Highlight key differences
3. Include market value insights
4. Give actionable recommendations

Make it engaging and friendly - like chatting with a knowledgeable friend! Use emojis and keep it fun! 🤝"""

        response = model.generate_content(prompt)
        print(f"[RAG] Response generated")
        return response.text
        
    except Exception as e:
        print(f"[RAG] Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, an error occurred: {str(e)}"

# Pre-load data
load_data()
