import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import unidecode
import threading

# Path to the user's CSV
CSV_PATH = os.path.join('moneyball_report_outputs', 'data chatbot.csv')

# Feature definitions (Reference Repo)
ATTACKER_FEATURES = [
    'Performance Gls', 'Performance Ast', 'Performance G+A',
    'Standard Sh', 'Standard SoT', 'Standard SoT%',
    'Standard Sh/90', 'Standard Dist',
    'Expected xG', 'Expected npxG', 'Expected xAG', 'Expected xA',
    'KP', 'PPA', 'CrsPA',
    'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
    'Aerial Duels Won', 'Aerial Duels Won%',
    'Tackles Att 3rd', 'Performance Off'
]

MIDFIELDER_FEATURES = [
    'Total Cmp', 'Total Att', 'KP', 'PPA', 'CrsPA',
    'Expected xA', 'Expected xAG',
    'Progression PrgP', 'Progression PrgC', 'Progression PrgR',
    'Total TotDist', 'Total PrgDist',
    'Int', 'Tkl+Int', 'Tackles Tkl', 'Tackles Mid 3rd',
    'Challenges Tkl%', 'Challenges Lost',
    'Performance Gls', 'Performance Ast',
    'Aerial Duels Won%', 'Performance Recov'
]

DEFENDER_FEATURES = [
    'Tackles Tkl', 'Tackles TklW', 'Int', 'Tkl+Int',
    'Challenges Tkl%', 'Challenges Lost',
    'Blocks Blocks', 'Blocks Sh', 'Blocks Pass',
    'Clr', 'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%',
    'Progression PrgP', 'Progression PrgC', 'Progression PrgR',
    'Total Cmp', 'Total Att', 'Total PrgDist',
    'Err'
]

GK_FEATURES = [
    'Performance GA', 'Performance SoTA', 'Performance Saves',
    'Performance Save%', 'Performance CS', 'Performance CS%',
    'Penalty Kicks PKsv', 'Penalty Kicks PKatt'
]

RADAR_CATEGORIES_DEFAULT = [
    'Performance Gls','Performance Ast','KP','GCA GCA',
    'Aerial Duels Won','Int','Tackles TklW',
    'Performance Saves','Performance CS','Performance GA','Performance SoTA'
]

ALL_FEATURES_BY_POSITION = {
    "attacker": ATTACKER_FEATURES,
    "midfielder": MIDFIELDER_FEATURES,
    "defender": DEFENDER_FEATURES,
    "goalkeeper": GK_FEATURES
}

# Mapping from Reference Column Names to User Column Names
COLUMN_MAPPING = {
    'Performance Gls': 'Goals',
    'Performance Gls': 'Assists',
    'Performance G+A': 'Goals + Assists',
    'Standard Sh': 'Sh',
    'Standard SoT': 'SoT',
    'Standard SoT%': 'SoT%',
    'Standard Sh/90': 'Sh/90',
    'Standard Dist': 'Dist',
    'Expected xG': 'xG',
    'Expected npxG': 'Non-Penalty xG',
    'Expected xAG': 'xAG',
    'Expected xA': 'xA',
    'KP': 'KP',
    'PPA': 'PPA',
    'CrsPA': 'CrsPA',
    'Progression PrgC': 'PrgC',
    'Progression PrgP': 'PrgP',
    'Progression PrgR': 'PrgR',
    'Aerial Duels Won': 'Won',
    'Aerial Duels Won%': 'Won%',
    'Tackles Att 3rd': 'Att 3rd',
    'Performance Off': 'Off_stats_misc', # Assuming Offsides
    'Total Cmp': 'Cmp',
    'Total Att': 'Att',
    'Total TotDist': 'TotDist',
    'Total PrgDist': 'PrgDist',
    'Int': 'Int',
    'Tkl+Int': 'Tkl+Int',
    'Tackles Tkl': 'Tkl',
    'Tackles Mid 3rd': 'Mid 3rd',
    'Challenges Tkl%': 'Tkl%', # Check this
    'Challenges Lost': 'Lost', # Check this
    'Performance Recov': 'Recov',
    'Tackles TklW': 'TklW',
    'Blocks Blocks': 'Blocks',
    'Blocks Sh': 'Sh_stats_defense', # Maybe?
    'Blocks Pass': 'Pass', # Maybe?
    'Clr': 'Clr',
    'Aerial Duels Lost': 'Lost_stats_misc', # Maybe?
    'Err': 'Err',
    'Performance GA': 'GA',
    'Performance SoTA': 'SoTA',
    'Performance Saves': 'Saves',
    'Performance Save%': 'Save%',
    'Performance CS': 'CS',
    'Performance CS%': 'CS%',
    'Penalty Kicks PKsv': 'PKsv',
    'Penalty Kicks PKatt': 'PKatt_stats_keeper',
    'GCA GCA': 'GCA',
    'Playing Time MP': 'Matches Played',
    'Born': 'Born',
    'Pos': 'Position',
    'Squad': 'Team',
    'Comp': 'league',
    'Age': 'Age',
    'Nation': 'Nation',
    'Player': 'Player',
    'Rk': 'Rk_stats_playing_time'
}

# Reverse mapping for display/logic if needed, or just use mapped columns
# We will rename user columns to reference columns when loading

FEATURE_DESCRIPTIONS = {
    "Rk": "Index or rank of the player in the list.",
    "Player": "Full name of the player.",
    "Nation": "Player's country of origin.",
    "Pos": "The position in which the player plays.",
    "Squad": "The team the player belongs to.",
    "Comp": "The competition in which the player participated.",
    "Age": "The player's age.",
    "Performance Gls": "Goals scored.",
    "Performance Ast": "Assists provided.",
    "Expected xG": "Expected Goals.",
    "Expected xA": "Expected Assists.",
    "KP": "Key Passes.",
    "Progression PrgC": "Progressive Carries.",
    "Progression PrgP": "Progressive Passes.",
    "Tackles Tkl": "Tackles made.",
    "Int": "Interceptions.",
    "Aerial Duels Won": "Aerial duels won.",
    # Add more as needed
}

_df_players: Optional[pd.DataFrame] = None
_lock = threading.Lock()

_pos_scalers: Dict[str, MinMaxScaler] = {}
_pos_feature_cols: Dict[str, List[str]] = {}
_pos_index_to_group_index: Dict[str, Dict[int, int]] = {}
_pos_group_matrices: Dict[str, np.ndarray] = {}
_pos_similarity: Dict[str, np.ndarray] = {}

def map_position_by_first(pos_raw: Any) -> str:
    if not isinstance(pos_raw, str):
        return "midfielder"
    token = pos_raw.split(",")[0].strip().upper()
    if token == "GK": return "goalkeeper"
    if token in ("FW","ST","CF","FW "): return "attacker"
    if token in ("MF","CM","CAM","AM","DM","CDM","RM","LM"): return "midfielder"
    if token in ("DF","CB","LB","RB","LWB","RWB"): return "defender"
    if "FW" in pos_raw: return "attacker"
    if "MF" in pos_raw: return "midfielder"
    if "DF" in pos_raw: return "defender"
    return "midfielder"

def _ensure_loaded():
    global _df_players, _pos_scalers, _pos_feature_cols, _pos_group_matrices, _pos_similarity, _pos_index_to_group_index
    with _lock:
        if _df_players is not None: return
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV not found at {CSV_PATH}.")
        
        df = pd.read_csv(CSV_PATH)
        
        # Rename columns based on mapping
        # Invert mapping to rename User -> Reference
        rename_map = {v: k for k, v in COLUMN_MAPPING.items() if v in df.columns}
        df.rename(columns=rename_map, inplace=True)
        
        df.columns = [c.strip() for c in df.columns]
        df.fillna(0, inplace=True)
        df['PlayerNormalized'] = df['Player'].astype(str).apply(lambda s: unidecode.unidecode(s).lower())
        
        if 'Rk' not in df.columns:
            df.insert(0, 'Rk', list(range(len(df))))
        else:
            try:
                df['Rk'] = df['Rk'].astype(int)
            except Exception:
                df['Rk'] = pd.to_numeric(df['Rk'], errors='coerce').fillna(-1).astype(int)
        
        df['PositionGroup'] = df['Pos'].apply(map_position_by_first)
        _df_players = df
        
        for group, feature_list in ALL_FEATURES_BY_POSITION.items():
            existing = [f for f in feature_list if f in df.columns]
            _pos_feature_cols[group] = existing
            if len(existing) == 0:
                _pos_scalers[group] = None
                _pos_group_matrices[group] = np.zeros((0, 0))
                _pos_similarity[group] = np.zeros((0, 0))
                _pos_index_to_group_index[group] = {}
                continue
            
            group_indices = df.index[df['PositionGroup'] == group].tolist()
            if len(group_indices) == 0:
                _pos_scalers[group] = None
                _pos_group_matrices[group] = np.zeros((0, len(existing)))
                _pos_similarity[group] = np.zeros((0, 0))
                _pos_index_to_group_index[group] = {}
                continue
                
            X = df.loc[group_indices, existing].copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.values)
            _pos_scalers[group] = scaler
            _pos_group_matrices[group] = X_scaled
            
            if X_scaled.shape[0] > 1:
                sim = cosine_similarity(X_scaled)
            else:
                sim = np.zeros((X_scaled.shape[0], X_scaled.shape[0]))
            _pos_similarity[group] = sim
            
            mapping = {int(idx): i for i, idx in enumerate(group_indices)}
            _pos_index_to_group_index[group] = mapping

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(v) for v in obj]
    if isinstance(obj, tuple): return tuple(clean(v) for v in obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def load_players_df() -> pd.DataFrame:
    _ensure_loaded()
    return _df_players.copy()

def search_players(q: str, rows: int = 20) -> List[Dict[str, Any]]:
    _ensure_loaded()
    if not q: return []
    qnorm = unidecode.unidecode(q).lower()
    df = _df_players[_df_players['PlayerNormalized'].str.contains(qnorm, na=False)]
    df = df.head(rows)
    return [{"player_id": int(r['Rk']), "player_name": r['Player']} for _, r in df.iterrows()]

def get_player_by_name_or_id(player_identifier: str) -> Optional[Dict[str, Any]]:
    _ensure_loaded()
    df = _df_players
    row = pd.DataFrame()
    try:
        pid = int(player_identifier); row = df.loc[df['Rk'] == pid]
    except Exception:
        needle = unidecode.unidecode(str(player_identifier)).lower()
        row = df[df['PlayerNormalized'] == needle]
        if row.empty:
            row = df[df['PlayerNormalized'].str.contains(needle, na=False)]
    if row.empty: return None
    r = row.iloc[0].to_dict()
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.integer,)): out[k] = int(v)
        elif isinstance(v, (np.floating,)): out[k] = float(v)
        else: out[k] = v
    return out

def _build_radar_for_player_row(row_index: int, category_labels: List[str]) -> Dict[str, Any]:
    _ensure_loaded()
    df = _df_players
    if row_index not in df.index: raise ValueError(f"Row index {row_index} not found")
    row = df.loc[row_index]
    player_pos_group = row['PositionGroup']
    labels = [c for c in category_labels if c in df.columns]
    values = []
    for label in labels:
        raw = 0.0
        try: raw = float(row.get(label, 0.0))
        except Exception: raw = 0.0
        
        pos_cols = _pos_feature_cols.get(player_pos_group, [])
        if label in pos_cols and _pos_scalers.get(player_pos_group) is not None:
            idx = pos_cols.index(label); scaler = _pos_scalers[player_pos_group]
            try:
                minv = float(scaler.data_min_[idx]); maxv = float(scaler.data_max_[idx])
                scaled = 0.0 if maxv == minv else (raw - minv) / (maxv - minv)
            except Exception:
                colmin = float(df[label].min()) if label in df.columns else 0.0
                colmax = float(df[label].max()) if label in df.columns else 1.0
                scaled = 0.0 if colmax == colmin else (raw - colmin) / (colmax - colmin)
        else:
            if label in df.columns:
                colmin = float(df[label].min()); colmax = float(df[label].max())
                scaled = 0.0 if colmax == colmin else (raw - colmin) / (colmax - colmin)
            else:
                scaled = 0.0
        scaled = max(0.0, min(1.0, float(scaled)))
        values.append(round(scaled, 4))
    return {"labels": labels, "values": values}

def get_player_stats_for_radar(player_identifier: str) -> Dict[str, Any]:
    _ensure_loaded()
    df = _df_players
    row_index = None
    try:
        pid = int(player_identifier)
        matches = df.loc[df['Rk'] == pid]
        if not matches.empty:
            row_index = matches.index[0]
    except Exception:
        needle = unidecode.unidecode(str(player_identifier)).lower()
        matches = df[df['PlayerNormalized'] == needle]
        if matches.empty:
            matches = df[df['PlayerNormalized'].str.contains(needle, na=False)]
        if matches.empty:
            raise ValueError(f"Player not found: {player_identifier}")
        row_index = matches.index[0]
    radar = _build_radar_for_player_row(row_index, RADAR_CATEGORIES_DEFAULT)
    return clean(radar)

def _attempt_group_for_player_index(global_index: int) -> str:
    _ensure_loaded()
    df = _df_players
    if global_index not in df.index: raise ValueError(f"Index {global_index} not found in df")
    primary = df.loc[global_index, 'PositionGroup']
    sim = _pos_similarity.get(primary, None)
    if isinstance(sim, np.ndarray) and sim.size > 1: return primary
    return primary

def _normalize_filter_param(param):
    if param is None: return None
    if isinstance(param, (list, tuple)):
        out = []
        for item in param:
            if not item: continue
            if isinstance(item, str) and ',' in item:
                out.extend([x.strip().lower() for x in item.split(',') if x.strip()])
            else:
                out.append(str(item).strip().lower())
        return out
    if isinstance(param, str):
        if ',' in param: return [x.strip().lower() for x in param.split(',') if x.strip()]
        return [param.strip().lower()]
    return None

def get_similar_players(player_id: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    _ensure_loaded()
    df = _df_players
    matches = None
    try:
        pid = int(player_id)
        matches = df.loc[df['Rk'] == pid]
    except Exception:
        needle = unidecode.unidecode(str(player_id)).lower()
        matches = df[df['PlayerNormalized'] == needle]
        if matches.empty:
            matches = df[df['PlayerNormalized'].str.contains(needle, na=False)]
    if matches is None or matches.empty:
        raise ValueError(f"Player not found: {player_id}")
    global_idx = int(matches.index[0])
    chosen_group = _attempt_group_for_player_index(global_idx)

    candidate_indices = list(df.index)
    if filters:
        cands = []
        min_age = filters.get("min_age")
        max_age = filters.get("max_age")
        leagues = _normalize_filter_param(filters.get("leagues"))
        positions = _normalize_filter_param(filters.get("positions"))
        
        for idx in df.index:
            row = df.loc[idx]
            ok = True
            if min_age is not None:
                try:
                    if float(row.get("Age", 0)) < float(min_age): ok = False
                except Exception: pass
            if max_age is not None:
                try:
                    if float(row.get("Age", 0)) > float(max_age): ok = False
                except Exception: pass
            if leagues:
                comp = str(row.get("Comp", "")).lower(); squad = str(row.get("Squad", "")).lower()
                if not any(l in comp or l in squad for l in leagues): ok = False
            if positions:
                posval = str(row.get("Pos", "")).lower()
                if not any(p == posval.split(",")[0].strip() or p in posval for p in positions): ok = False
            if ok: cands.append(idx)
        candidate_indices = cands

    mapping = _pos_index_to_group_index.get(chosen_group, {})
    group_candidate_pairs = []
    for idx in candidate_indices:
        if idx == global_idx: continue
        if idx in mapping:
            group_candidate_pairs.append((idx, mapping[idx]))
    
    if len(group_candidate_pairs) == 0: return []

    query_group_index = _pos_index_to_group_index.get(chosen_group, {}).get(global_idx, None)
    results = []
    sim_matrix = _pos_similarity.get(chosen_group, None)
    
    if query_group_index is not None and isinstance(sim_matrix, np.ndarray) and sim_matrix.size > 0:
        sims = sim_matrix[query_group_index]
        pairs = []
        for (gidx, grp_idx) in group_candidate_pairs:
            if grp_idx < len(sims):
                score = float(sims[grp_idx])
                pairs.append((gidx, score))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    else:
        return []

    top_pairs = pairs_sorted[:top_k]
    for gidx, score in top_pairs:
        row = df.loc[gidx]
        top_stats = {}
        for c in RADAR_CATEGORIES_DEFAULT:
            if c in df.columns:
                try: top_stats[c] = float(row.get(c, 0.0))
                except Exception: top_stats[c] = 0.0
        radar = _build_radar_for_player_row(gidx, RADAR_CATEGORIES_DEFAULT)
        results.append({
            "Rk": int(row.get("Rk", int(gidx))),
            "Player": row.get("Player", ""),
            "Pos": row.get("Pos", ""),
            "Squad": row.get("Squad", ""),
            "Age": row.get("Age", ""),
            "Nation": row.get("Nation", ""),
            "similarity_score": float(score),
            "top_stats": top_stats,
            "radar": radar
        })

    return clean(results)

