from dotenv import load_dotenv
import os, json

# load .env from project root
load_dotenv()
print('API_FOOTBALL_KEY present:', bool(os.getenv('API_FOOTBALL_KEY')))

try:
    import live_api
    res = live_api.get_player_live_summary('Mason Greenwood')
    print('Result (short):')
    if res:
        print(json.dumps(res, default=str, indent=2))
    else:
        print('No result')
except Exception as e:
    print('Error calling live_api:', e)
