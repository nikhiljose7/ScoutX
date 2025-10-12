from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Global variables for models and data
models = {}
data = None

def load_data():
    """Load the processed player data"""
    global data
    try:
        data = pd.read_csv('players_data_with_weights.csv')
        return True
    except:
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
def chatbot():
    """Chatbot module"""
    return render_template('chatbot.html')

@app.route('/api/undervalued', methods=['POST'])
def get_undervalued():
    """API endpoint for undervalued players"""
    try:
        position = request.json.get('position', 'ALL')
        top_n = request.json.get('top_n', 25)
        
        # Load results from the analysis
        results_df = pd.read_csv('moneyball_report_outputs/all_predictions_with_undervaluation.csv')
        
        if position != 'ALL':
            results_df = results_df[results_df['Model_Pos'] == position]
        
        top_players = results_df.sort_values('Undervaluation', ascending=False).head(top_n)
        
        return jsonify({
            'success': True,
            'data': top_players.to_dict('records')
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
        
        player_data = data[data['Player'] == player_name].to_dict('records')
        
        if len(player_data) > 0:
            return jsonify({
                'success': True,
                'data': player_data[0]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Player not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_data()
    app.run(debug=True)