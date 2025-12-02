from dotenv import load_dotenv
load_dotenv()

# Monkeypatch for GenerationConfig.MediaResolution error
import google.generativeai.types as types
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

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from chatbot import chatbot_bp
import similarity_service
import gemini_service

load_dotenv()

app = Flask(__name__)
# Add a secret key for session management
app.secret_key = os.urandom(24)
app.register_blueprint(chatbot_bp)

# Global variables for models and data
models = {}
data = None

def load_data():
    """Load the processed player data"""
    global data
    # Try multiple possible data sources so the API is resilient when one file is missing
    # Prefer the cleaned players dataset (with market values) when available
    candidates = [
        'players_data_with_weights.csv',
        os.path.join('moneyball_report_outputs', 'players_data_cleaned_with_market_values_with_market_values1.csv'),
        os.path.join('moneyball_report_outputs', 'all_predictions_with_undervaluation.csv')
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                data = pd.read_csv(path)
                # Normalize Player column to string
                if 'Player' in data.columns:
                    data['Player'] = data['Player'].astype(str)
                return True
        except Exception as e:
            # try next candidate
            continue

    return False

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/undervalued')
def undervalued():
    """Undervalued players module"""
    return render_template('undervalued.html')

@app.route('/similar')
def similar():
    """Similar player identification module"""
    return render_template('similar.html')

@app.route('/comparison')
def comparison():
    """Player comparison module"""
    return render_template('comparison.html')

@app.route('/chatbot')
def chatbot_page():
    """Chatbot module page"""
    return render_template('chatbot.html')


@app.route('/player')
def player_page():
    """Player details page"""
    # Allow passing the player name via query string so the template can
    # render or fetch the details immediately. Keeps existing behavior
    # where the client-side script reads the query param as well.
    player_name = request.args.get('player', '')
    return render_template('player.html', player_name=player_name)

@app.route('/api/undervalued/filters', methods=['GET'])
def get_filter_options():
    """Get available filter options for the undervalued players page"""
    try:
        # Load results from the analysis
        results_df = pd.read_csv('moneyball_report_outputs/all_predictions_with_undervaluation.csv')
        
        return jsonify({
            'success': True,
            'leagues': results_df['Comp'].unique().tolist(),
            'squads': results_df['Squad'].unique().tolist()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/undervalued', methods=['POST'])
def get_undervalued():
    """API endpoint for undervalued players"""
    try:
        # Get filter parameters
        position = request.json.get('position', 'ALL')
        league = request.json.get('league', 'ALL')
        squad = request.json.get('squad', 'ALL')
        page = request.json.get('page', 1)
        items_per_page = request.json.get('items_per_page', 25)
        sort_column = request.json.get('sort_column', 'undervaluation')
        sort_direction = request.json.get('sort_direction', 'desc')
        min_age = request.json.get('min_age')
        max_age = request.json.get('max_age')
        min_value = request.json.get('min_value')
        max_value = request.json.get('max_value')
        min_undervaluation = request.json.get('min_undervaluation')
        
        # Load results from the analysis
        results_df = pd.read_csv('moneyball_report_outputs/all_predictions_with_undervaluation.csv')
        
        # Apply filters
        if position != 'ALL':
            results_df = results_df[results_df['Model_Pos'] == position]
            
        if league != 'ALL':
            results_df = results_df[results_df['Comp'] == league]
            
        if squad != 'ALL':
            results_df = results_df[results_df['Squad'] == squad]
            
        if min_age is not None:
            results_df = results_df[results_df['Age'] >= min_age]
            
        if max_age is not None:
            results_df = results_df[results_df['Age'] <= max_age]
            
        if min_value is not None:
            results_df = results_df[results_df['Market_Value_Million_EUR'] >= min_value]
            
        if max_value is not None:
            results_df = results_df[results_df['Market_Value_Million_EUR'] <= max_value]
            
        if min_undervaluation is not None:
            results_df = results_df[results_df['Undervaluation'] >= min_undervaluation]

        # Get total count before pagination
        total_items = len(results_df)

        # Map frontend column names to DataFrame columns
        column_mapping = {
            'rank': 'Undervaluation',  # Default to undervaluation for rank
            'player': 'Player',
            'position': 'Main_Pos',
            'squad': 'Squad',
            'league': 'Comp',
            'age': 'Age',
            'market_value': 'Market_Value_Million_EUR',
            'predicted_value': 'Predicted_Value',
            'undervaluation': 'Undervaluation'  # Make sure this matches the column name exactly
        }

        # Ensure numeric columns are properly typed
        numeric_columns = ['Age', 'Market_Value_Million_EUR', 'Predicted_Value', 'Undervaluation']
        for col in numeric_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        # Sort the data
        sort_col = column_mapping.get(sort_column, 'Undervaluation')
        ascending = sort_direction == 'asc'
        print(f"Sorting by column: {sort_col}, ascending: {ascending}")  # Debug log
        
        # Ensure the column exists
        if sort_col not in results_df.columns:
            print(f"Warning: Column {sort_col} not found in DataFrame. Available columns: {results_df.columns.tolist()}")
            sort_col = 'Undervaluation'  # Fallback to default
            
        results_df = results_df.sort_values(sort_col, ascending=ascending)
        
        # Apply pagination
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = results_df.iloc[start_idx:end_idx]
        
        return jsonify({
            'success': True,
            'data': page_data.to_dict('records'),
            'total_items': total_items
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of all players"""
    try:
        if data is None:
            load_data()
        
        players = data[['Player', 'Squad', 'Comp', 'Pos']].to_dict('records')
        return jsonify({
            'success': True,
            'data': players
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/player/<player_name>', methods=['GET'])
def get_player_details(player_name):
    """Get details for a specific player"""
    try:
        if data is None:
            load_data()
        # Perform case-insensitive exact match first
        if 'Player' not in data.columns:
            return jsonify({'success': False, 'error': 'Player column not available in data'})

        player_name_norm = str(player_name).strip().lower()
        players_series = data['Player'].astype(str).str.strip()
        mask_exact = players_series.str.lower() == player_name_norm
        matches = data[mask_exact]

        # If no exact match, try a case-insensitive substring match (first hit)
        if matches.empty:
            mask_contains = players_series.str.lower().str.contains(player_name_norm, na=False)
            matches = data[mask_contains]

        if not matches.empty:
            # return the first matching record as a dict and sanitize values
            player_series = matches.iloc[0]

            def _sanitize_value(v):
                # Replace pandas/np NA/NaN/NaT with None so JSON uses null
                try:
                    if pd.isna(v):
                        return None
                except Exception:
                    pass

                # Convert numpy scalar types to native Python types
                if isinstance(v, (np.generic,)):
                    try:
                        return v.item()
                    except Exception:
                        pass

                # Convert pandas Timestamp to ISO string
                try:
                    if isinstance(v, pd.Timestamp):
                        return v.isoformat()
                except Exception:
                    pass

                # Convert infinities to None
                try:
                    if isinstance(v, float) and not math.isfinite(v):
                        return None
                except Exception:
                    pass

                return v

            player_data = {k: _sanitize_value(v) for k, v in player_series.to_dict().items()}
            return jsonify({'success': True, 'data': player_data})

        return jsonify({'success': False, 'error': 'Player not found'})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/compare', methods=['POST'])
def compare_players_api():
    """Compare players by names provided in JSON payload: { "players": ["name1","name2"] }
    Returns a list of player data objects (or placeholders) in the same order.
    """
    try:
        payload = request.json or {}
        names = payload.get('players') or []
        if not isinstance(names, list) or len(names) < 2:
            return jsonify({'success': False, 'error': 'Provide at least two player names in "players" list'}), 400

        if data is None:
            load_data()

        def _sanitize_value(v):
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if isinstance(v, (np.generic,)):
                try:
                    return v.item()
                except Exception:
                    pass
            try:
                if isinstance(v, pd.Timestamp):
                    return v.isoformat()
            except Exception:
                pass
            try:
                if isinstance(v, float) and not math.isfinite(v):
                    return None
            except Exception:
                pass
            return v

        results = []
        players_series = data['Player'].astype(str).str.strip() if 'Player' in data.columns else pd.Series([])

        for name in names:
            name_norm = str(name).strip().lower()
            found = None
            if 'Player' in data.columns:
                mask_exact = players_series.str.lower() == name_norm
                matches = data[mask_exact]
                if matches.empty:
                    mask_contains = players_series.str.lower().str.contains(name_norm, na=False)
                    matches = data[mask_contains]
                if not matches.empty:
                    player_series = matches.iloc[0]
                    player_data = {k: _sanitize_value(v) for k, v in player_series.to_dict().items()}
                    found = player_data

            if not found:
                # placeholder with requested name
                found = {'Player': name, 'note': 'Not found in dataset'}

            results.append(found)

        return jsonify({'success': True, 'data': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/meta', methods=['GET'])
def api_meta():
    """Get metadata for filters (leagues, positions)"""
    try:
        similarity_service._ensure_loaded()
        df = similarity_service._df_players
        leagues = sorted(df["Comp"].dropna().astype(str).unique().tolist()) if "Comp" in df else []
        pos_set = set()
        if "Pos" in df.columns:
            for raw in df["Pos"].dropna().astype(str).unique():
                parts = [p.strip() for p in raw.split(',')]
                for p in parts:
                    pos_set.add(p)
        return jsonify({"ok": True, "leagues": leagues, "positions": sorted(list(pos_set))})
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)})

@app.route('/api/feature_desc', methods=['GET'])
def api_feature_desc():
    """Get feature descriptions"""
    return jsonify({"ok": True, "descriptions": similarity_service.FEATURE_DESCRIPTIONS})

@app.route('/api/search', methods=['GET'])
def api_search():
    """Search players by name"""
    q = request.args.get('q', '')
    rows = int(request.args.get('rows', 20))
    try:
        res = similarity_service.search_players(q, rows)
        return jsonify({"ok": True, "results": res})
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)})

@app.route('/api/player_details', methods=['GET'])
def api_player_details_new():
    """Get player details by ID or Name (for Similar Players page)"""
    player_id = request.args.get('player_id')
    try:
        p = similarity_service.get_player_by_name_or_id(player_id)
        if not p:
            return jsonify({"ok": False, "detail": "Player not found"}), 404
        return jsonify({"ok": True, "player": similarity_service.clean(p)})
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)})

@app.route('/api/similar_players', methods=['GET'])
def api_similar_players():
    """Get similar players"""
    player_id = request.args.get('player_id')
    k = int(request.args.get('k', 10))
    min_age = request.args.get('min_age')
    max_age = request.args.get('max_age')
    leagues = request.args.get('leagues')
    positions = request.args.get('positions')
    
    filters = {
        "min_age": min_age,
        "max_age": max_age,
        "leagues": leagues,
        "positions": positions
    }
    
    try:
        sim = similarity_service.get_similar_players(player_id, top_k=k, filters=filters)
        rad = similarity_service.get_player_stats_for_radar(player_id)
        return jsonify(similarity_service.clean({"ok": True, "results": sim, "input_radar": rad}))
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)})

@app.route('/api/compare_players', methods=['POST'])
def api_compare_players_new():
    """Compare multiple players and generate AI report"""
    try:
        payload = request.json or {}
        player_ids = payload.get('player_ids', [])
        
        if len(player_ids) < 2:
            return jsonify({"ok": False, "detail": "At least 2 players required."}), 400
            
        players = []
        for pid in player_ids:
            p = similarity_service.get_player_by_name_or_id(pid)
            if not p:
                return jsonify({"ok": False, "detail": f"Player not found: {pid}"}), 404
            players.append(similarity_service.clean(p))
            
        radars = [similarity_service.get_player_stats_for_radar(p.get("Rk") or p.get("Player")) for p in players]
        keys = ["Player", "Pos", "Squad", "Age"] + radars[0]["labels"]
        rows = []
        for p in players:
            row = {k: p.get(k, "") for k in keys}
            rows.append({"Rk": p.get("Rk"), "stats": row})
            
        ai_payload = {"players": players, "radar": radars}
        ai_report = gemini_service.generate_comparison_report(ai_payload)
        
        return jsonify(similarity_service.clean({
            "ok": True,
            "players": players,
            "radar": radars,
            "compare_stats": {"keys": keys, "rows": rows},
            "ai_report": ai_report
        }))
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)})


if __name__ == '__main__':
    load_data()
    app.run(debug=True)