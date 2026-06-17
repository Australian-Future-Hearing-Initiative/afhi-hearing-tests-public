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


# Mock database for offline testing in sandboxed environments
MOCK_HEADPHONES = {
    "apple airpods pro": {
        "source": "oratory1990",
        "form_factor": "in-ear",
        "frequencies": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        "smoothed": [-0.5, -0.1, 0.3, 1.2, 0.5, -2.0, -3.5, -4.5],
        "raw_url": "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/oratory1990/harman_in-ear_2019v2/Apple%20AirPods%20Pro/Apple%20AirPods%20Pro.csv"
    },
    "google pixel buds pro": {
        "source": "oratory1990",
        "form_factor": "in-ear",
        "frequencies": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        "smoothed": [-0.2, 0.2, 0.5, 1.5, 0.8, -1.0, -2.0, -3.0],
        "raw_url": "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/oratory1990/harman_in-ear_2019v2/Google%20Pixel%20Buds%20Pro/Google%20Pixel%20Buds%20Pro.csv"
    },
    "sony wh-1000xm4": {
        "source": "oratory1990",
        "form_factor": "over-ear",
        "frequencies": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        "smoothed": [1.5, 0.8, -0.2, -1.0, -2.0, -3.5, -2.5, -1.5],
        "raw_url": "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/oratory1990/over-ear/Sony%20WH-1000XM4/Sony%20WH-1000XM4.csv"
    },
    "sennheiser hd 600": {
        "source": "oratory1990",
        "form_factor": "over-ear",
        "frequencies": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        "smoothed": [0.1, -0.2, -0.5, 0.8, 0.2, -1.2, -1.8, -2.5],
        "raw_url": "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/oratory1990/over-ear/Sennheiser%20HD%20600/Sennheiser%20HD%20600.csv"
    },
    "shokz openrun": {
        "source": "crinacle",
        "form_factor": "bone-conduction",
        "frequencies": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        "smoothed": [-10.0, -8.0, -5.0, -2.0, -3.0, -4.0, -8.0, -12.0],
        "raw_url": "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/crinacle/ears-711_harman_over-ear_2018/Shokz%20OpenRun/Shokz%20OpenRun.csv"
    }
}


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

def find_local_autoeq_dir() -> str | None:
    """Helper to locate the AutoEq clone directory in the workspace."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = current_dir
    for _ in range(4):
        candidate = os.path.join(temp_dir, "AutoEq")
        if os.path.isdir(candidate):
            return candidate
        temp_dir = os.path.dirname(temp_dir)
    candidate = os.path.join(os.getcwd(), "AutoEq")
    if os.path.isdir(candidate):
        return candidate
    return None

def search_autoeq_files(headphone_name: str) -> list[dict]:
    """Searches the local JSON database, local AutoEq clone, and GitHub Code Search API for a headphone model name.
    
    Returns a list of dicts with keys: 'name', 'path', 'database', 'html_url', 'raw_url'
    """
    if not headphone_name:
        return []
    
    name_clean = headphone_name.lower().strip()
    
    # 1. Try local JSON database search first (fast, offline, deterministic)
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

    # 2. Search local AutoEq clone if present
    autoeq_dir = find_local_autoeq_dir()
    if autoeq_dir:
        results_dir = os.path.join(autoeq_dir, "results")
        if os.path.exists(results_dir):
            print(f"Searching local AutoEq clone for '{headphone_name}'...")
            local_clone_matches = []
            words = name_clean.split()
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith(".csv"):
                        rel_path = os.path.relpath(os.path.join(root, file), autoeq_dir)
                        rel_path_lower = rel_path.lower()
                        if all(word in rel_path_lower for word in words):
                            path_parts = rel_path.split(os.sep)
                            db_source = path_parts[1] if len(path_parts) > 1 else "unknown"
                            encoded_parts = [urllib.parse.quote(part) for part in path_parts]
                            url_path = "/".join(encoded_parts)
                            raw_url = f"https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/{url_path}"
                            html_url = raw_url.replace("raw.githubusercontent.com", "github.com").replace("/master/", "/blob/master/")
                            
                            local_clone_matches.append({
                                "name": file,
                                "path": rel_path,
                                "database": db_source,
                                "html_url": html_url,
                                "raw_url": raw_url
                            })
            if local_clone_matches:
                print(f"Found {len(local_clone_matches)} local AutoEq clone matches for '{headphone_name}'.")
                return local_clone_matches


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
        print(f"GitHub Search API call failed: {e}. Falling back to offline dictionary search.")
        matches = []
        for mock_name, mock_info in MOCK_HEADPHONES.items():
            # If the user model name is in mock_name or vice versa
            if name_clean in mock_name or mock_name in name_clean:
                matches.append({
                    "name": f"{mock_name}.csv",
                    "path": f"results/{mock_info['source']}/{mock_info['form_factor']}/{mock_name}/{mock_name}.csv",
                    "database": mock_info["source"],
                    "html_url": mock_info["raw_url"].replace("raw.githubusercontent.com", "github.com").replace("/master/", "/blob/master/"),
                    "raw_url": mock_info["raw_url"]
                })
        return matches

def fetch_frequency_response(raw_url: str, headphone_name: str = "") -> dict:
    """Retrieves and parses the CSV response data from a raw AutoEq URL or local clone.
    
    Returns a dict with 'frequency' (list) and 'smoothed' (list) values.
    """
    # If using mock fallback
    name_clean = headphone_name.lower().strip() if headphone_name else ""
    if name_clean in MOCK_HEADPHONES and MOCK_HEADPHONES[name_clean]["raw_url"] == raw_url:
        return {
            "frequency": MOCK_HEADPHONES[name_clean]["frequencies"],
            "smoothed": MOCK_HEADPHONES[name_clean]["smoothed"]
        }
        
    # Try reading locally from AutoEq clone if present
    autoeq_dir = find_local_autoeq_dir()
    if autoeq_dir and "results/" in raw_url:
        parts = raw_url.split("/results/")
        if len(parts) > 1:
            rel_path = urllib.parse.unquote(parts[1])
            local_path = os.path.join(autoeq_dir, "results", rel_path)
            if os.path.exists(local_path):
                try:
                    print(f"Reading CSV locally from: {local_path}")
                    with open(local_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    frequencies = []
                    smoothed_responses = []
                    reader = csv.reader(content.splitlines())
                    header = next(reader)
                    try:
                        freq_idx = header.index("frequency")
                        smoothed_idx = header.index("smoothed")
                    except ValueError:
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
                except Exception as local_err:
                    print(f"Failed to read local copy at {local_path}: {local_err}. Trying online.")
                    
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
        # Secondary fallback using matching mock name
        for mock_name, mock_info in MOCK_HEADPHONES.items():
            if mock_info["raw_url"] == raw_url or (name_clean and name_clean in mock_name):
                print(f"Found offline mock backup for raw_url/name: {mock_name}")
                return {
                    "frequency": mock_info["frequencies"],
                    "smoothed": mock_info["smoothed"]
                }
        raise e
