import os
import urllib.request
import urllib.parse
import json
import csv
import re
from .config import BONE_CONDUCTION_KEYWORDS

# Load local headphone database
dir_path = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(dir_path, "headphone_database.json")

try:
    with open(db_path, "r") as f:
        LOCAL_DATABASE = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load local headphone database from {db_path}: {e}")
    LOCAL_DATABASE = {}

def is_bone_conduction_device(name: str) -> bool:
    """Checks if a headphone model name indicates bone conduction."""
    if not name:
        return False
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in BONE_CONDUCTION_KEYWORDS)

def github_to_raw_url(html_url: str) -> str:
    """Converts a standard GitHub file URL to a raw content URL."""
    # Example: https://github.com/jaakkopasanen/AutoEq/blob/master/results/oratory1990/...
    # -> https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/oratory1990/...
    return html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

def search_autoeq_files(headphone_name: str) -> list[dict]:
    """Searches the local JSON database, and GitHub Code Search API for a headphone model name.
    
    Returns a list of dicts with keys: 'name', 'path', 'database', 'html_url', 'raw_url'
    """
    if not headphone_name:
        return []
    
    name_clean = headphone_name.lower().strip()
    
    # Lookup model in local JSON database 
    local_matches = []
    for key, info in LOCAL_DATABASE.items():
        if name_clean == key or name_clean in key or key in name_clean:
            # Reconstruct model name without source suffix for 'name' field
            base_name = re.sub(r"\s*\([^)]+\)$", "", key).strip()
            
            # Reconstruct paths
            raw_url = info["raw_url"]
            path = raw_url.split("AutoEq/master/")[1]
            html_url = raw_url.replace("raw.githubusercontent.com", "github.com").replace("/master/", "/blob/master/")
            
            local_matches.append({
                "name": f"{base_name} ({info['source']})",
                "path": path,
                "database": info["source"],
                "html_url": html_url,
                "raw_url": raw_url
            })
            
    if local_matches:
        print(f"Deterministic Match Found locally for '{headphone_name}': {len(local_matches)} result(s).")
        return local_matches



    # Format the query for GitHub Code Search API
    # Since search code is constrained to files matching the name
    query_str = f"filename:\"{headphone_name}\" extension:csv repo:jaakkopasanen/AutoEq path:results"
    encoded_query = urllib.parse.quote(query_str)
    api_url = f"https://api.github.com/search/code?q={encoded_query}"
    
    headers = {
        "User-Agent": "HearingTestCalibrationAgent/1.0",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            
            results = []
            for item in items:
                path = item.get("path", "")
                # Extract database source from path (results/<source>/...)
                path_parts = path.split("/")
                db_source = path_parts[1] if len(path_parts) > 1 else "unknown"
                
                results.append({
                    "name": item.get("name", ""),
                    "path": path,
                    "database": db_source,
                    "html_url": item.get("html_url", ""),
                    "raw_url": github_to_raw_url(item.get("html_url", ""))
                })
            return results
            
    except Exception as e:
        # Fallback to local mock database if network is offline/unavailable
        print(f"GitHub Search API call failed: {e}.")
        
def fetch_frequency_response(raw_url: str, headphone_name: str = "") -> dict:
    """Retrieves and parses the CSV response data from a raw AutoEq URL.
    
    Returns a dict with 'frequency' (list) and 'smoothed' (list) values.
    """
                    
    headers = {
        "User-Agent": "HearingTestCalibrationAgent/1.0"
    }
    
    try:
        req = urllib.request.Request(raw_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode('utf-8')
            
            frequencies = []
            smoothed_responses = []
            
            reader = csv.reader(content.splitlines())
            header = next(reader)
            
            # Find column indices
            try:
                freq_idx = header.index("frequency")
                smoothed_idx = header.index("smoothed")
            except ValueError:
                # If column names differ, fall back to indices 0 and 2
                freq_idx, smoothed_idx = 0, 2
                
            for row in reader:
                if len(row) > max(freq_idx, smoothed_idx):
                    try:
                        frequencies.append(float(row[freq_idx]))
                        smoothed_responses.append(float(row[smoothed_idx]))
                    except ValueError:
                        continue
                        
            return {
                "frequency": frequencies,
                "smoothed": smoothed_responses
            }
            
    except Exception as e:
        print(f"Failed to fetch CSV data from {raw_url}: {e}")
        raise e
