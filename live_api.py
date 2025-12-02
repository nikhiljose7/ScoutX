import os
import requests
from typing import Optional, Dict, Any, List

# API-Football helper. Uses api-sports v3 (https://v3.football.api-sports.io).
# Provide API key via the environment variable API_FOOTBALL_KEY.

API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
API_FOOTBALL_BASE = os.getenv('API_FOOTBALL_BASE', 'https://v3.football.api-sports.io')


def _call_api(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Call api-football and return parsed JSON or None on error."""
    if not API_FOOTBALL_KEY:
        # No key configured
        print('[live_api] API_FOOTBALL_KEY not configured')
        return None
    url = f"{API_FOOTBALL_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = {'x-apisports-key': API_FOOTBALL_KEY}
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
        if resp.status_code != 200:
            print(f"[live_api] api-football returned status {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except Exception:
        import traceback
        print('[live_api] Exception when calling api-football:')
        traceback.print_exc()
        return None


def get_player_live_summary(name: str) -> Optional[Dict[str, Any]]:
    """Search the player by name and return a concise summary dict or None.

    Returned dict example:
      {
        'name': 'Full Name',
        'nationality': 'Country',
        'age': 28,
        'photo': 'https://...',
        'team': 'Team Name',
        'league_stats': [...],
        'source': 'api-football'
      }
    """
    # The API requires league and season for player search.
    # We'll try Premier League (39) and Champions League (2) for recent seasons.
    # This is a heuristic since we don't know the player's league.
    common_leagues = [39, 2, 140, 78, 135, 61] # PL, CL, La Liga, Bundesliga, Serie A, Ligue 1
    seasons = [2024, 2023]
    
    for season in seasons:
        for league in common_leagues:
            j = _call_api('players', {'search': name, 'league': league, 'season': season})
            if j and 'response' in j:
                items = j.get('response') or []
                if items:
                    # Found a match
                    first = items[0]
                    player = first.get('player') or {}
                    statistics = first.get('statistics') or []

                    summary = {
                        'name': player.get('name'),
                        'nationality': player.get('nationality'),
                        'age': player.get('age') if 'age' in player else None,
                        'photo': player.get('photo'),
                        'team': (statistics[0].get('team', {}).get('name')) if statistics else None,
                        'league_stats': statistics,
                        'source': 'api-football'
                    }
                    return summary
    return None


def get_live_matches_summary(limit: int = 5) -> Optional[str]:
    """Get a short textual summary of currently live fixtures (if any).

    Returns a newline-delimited string or None if API not configured.
    """
    j = _call_api('fixtures', {'live': 'all'})
    if not j or 'response' not in j:
        return None
    matches = j.get('response') or []
    if not matches:
        return "There are no live matches currently."
    summaries: List[str] = []
    for match in matches[:limit]:
        fixture = match.get('fixture', {})
        teams = match.get('teams', {})
        goals = match.get('goals', {})
        home = teams.get('home', {}).get('name', 'N/A')
        away = teams.get('away', {}).get('name', 'N/A')
        home_goals = goals.get('home')
        away_goals = goals.get('away')
        score_line = f"{home_goals} - {away_goals}" if home_goals is not None else "N/A"
        status = fixture.get('status', {}).get('short', 'N/A')
        elapsed = fixture.get('status', {}).get('elapsed')
        elapsed_str = f"{elapsed}'" if elapsed is not None else ''
        summaries.append(f"{home} {score_line} {away} ({elapsed_str}, {status})")
    return "\n".join(summaries)


def get_team_recent_matches(team_name: str, last: int = 5) -> Optional[str]:
    """Return a short summary of recent matches for a team (uses teams + fixtures endpoints)."""
    # Find team
    j_team = _call_api('teams', {'search': team_name})
    if not j_team or 'response' not in j_team:
        return None
    resp = j_team.get('response') or []
    if not resp:
        return None
    team_obj = resp[0].get('team') or {}
    team_id = team_obj.get('id')
    found_name = team_obj.get('name')
    if not team_id:
        return None

    # Free plan limitation: 'last' parameter is not available.
    # Workaround: Fetch matches for the current/recent season and filter locally.
    # We'll try 2024 first (covering 2024-2025), then 2023 if needed.
    # Note: In a production app, we should dynamically determine the current season.
    matches = []
    for season in [2025, 2024, 2023]:
        j_fixtures = _call_api('fixtures', {'team': team_id, 'season': season})
        if j_fixtures and 'response' in j_fixtures:
            season_matches = j_fixtures.get('response') or []
            if season_matches:
                matches = season_matches
                break
    
    if not matches:
        return f"No recent matches found for {found_name} (checked seasons 2025, 2024)."

    # Filter for finished matches only
    finished_matches = [m for m in matches if m.get('fixture', {}).get('status', {}).get('short') in ['FT', 'AET', 'PEN']]
    
    # Sort by date descending
    finished_matches.sort(key=lambda x: x.get('fixture', {}).get('date', ''), reverse=True)
    
    # Take the last N
    recent = finished_matches[:last]

    summaries: List[str] = []
    for m in recent:
        fixture = m.get('fixture', {})
        teams = m.get('teams', {})
        goals = m.get('goals', {})
        date = fixture.get('date', 'N/A').split('T')[0]
        home = teams.get('home', {}).get('name', 'N/A')
        away = teams.get('away', {}).get('name', 'N/A')
        home_goals = goals.get('home')
        away_goals = goals.get('away')
        score_line = f"{home} {home_goals} - {away_goals} {away}"
        summaries.append(f"Date: {date} | {score_line}")
    
    return f"Recent {len(summaries)} matches for {found_name}:\n" + "\n".join(summaries)


# ---- SofaScore via RapidAPI integration ----
# Configure via environment variables to avoid hardcoding keys in source.
SOFASCORE_RAPIDAPI_KEY = os.getenv('SOFASCORE_RAPIDAPI_KEY')
SOFASCORE_RAPIDAPI_HOST = os.getenv('SOFASCORE_RAPIDAPI_HOST', 'sofascore.p.rapidapi.com')
SOFASCORE_BASE = os.getenv('SOFASCORE_BASE', 'https://sofascore.p.rapidapi.com')


def _call_sofascore(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Call the SofaScore RapidAPI proxy and return parsed JSON or None on error.

    `path` should start with a slash (e.g. '/tvchannels/get-available-countries').
    The RapidAPI key must be provided in the environment variable SOFASCORE_RAPIDAPI_KEY.
    """
    if not SOFASCORE_RAPIDAPI_KEY:
        return None
    url = SOFASCORE_BASE.rstrip('/') + path
    headers = {
        'x-rapidapi-key': SOFASCORE_RAPIDAPI_KEY,
        'x-rapidapi-host': SOFASCORE_RAPIDAPI_HOST,
    }
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
        if resp.status_code != 200:
            return None
        # Attempt to parse JSON; SofaScore sometimes returns nested structures
        return resp.json()
    except Exception:
        return None


def get_match_tv_countries(match_id: int) -> Optional[Dict[str, Any]]:
    """Return available TV countries / channels for a given SofaScore match id.

    Returns the parsed JSON from SofaScore RapidAPI or None on error.
    """
    try:
        path = f"/tvchannels/get-available-countries"
        params = {'matchId': int(match_id)}
    except Exception:
        return None
    return _call_sofascore(path, params=params)
