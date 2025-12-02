import os
import google.generativeai as genai
from typing import Dict, Any

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

AI_PROMPT_TEMPLATE = """
You are an elite European football scouting analyst.
Compare the players strictly based on the provided data (stats + radar values).

Your analysis should:
- Identify technical strengths
- Identify weaknesses
- Compare each player directly to the others
- Highlight tactical suitability
- Mention role tendencies and expected usage
- Keep the summary concise (~300 words max)
"""

def generate_comparison_report(payload: Dict[str, Any]) -> str:
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found in environment variables."

    try:
        players = payload.get("players", [])
        
        lines = [AI_PROMPT_TEMPLATE.strip(), "\nPLAYER DATA:\n"]
        brief_keys = [
            "Performance Gls", "Performance Ast", "KP",
            "Expected xG", "Expected xA",
            "Int", "Tackles Tkl", "Aerial Duels Won"
        ]
        
        # Mapping for brief keys if they differ in payload
        # The payload comes from similarity_service which should have normalized keys or original keys
        # We'll try to find the keys or their mapped equivalents
        
        for p in players:
            name = p.get("Player", "Unknown")
            pos  = p.get("Pos", "")
            club = p.get("Squad", "")
            snippet = []
            
            # Helper to find value even if key is slightly different
            def get_val(obj, key):
                if key in obj: return obj[key]
                # Try simple mapping
                simple_key = key.replace("Performance ", "").replace("Expected ", "").replace("Standard ", "")
                if simple_key in obj: return obj[simple_key]
                return None

            for k in brief_keys:
                val = get_val(p, k)
                if val is not None:
                    snippet.append(f"{k}: {val}")
            
            lines.append(f"- {name} | {pos} | {club} | " + "; ".join(snippet))
        
        lines.append("\nProvide a direct comparison and a summary.\n")
        prompt_text = "\n".join(lines)

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt_text)
        
        return response.text
    except Exception as e:
        return f"Error generating report: {str(e)}"
