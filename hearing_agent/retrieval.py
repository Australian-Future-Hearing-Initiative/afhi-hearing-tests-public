import os
import urllib.request
import urllib.parse
import json
import re
import tempfile
from autoeq.frequency_response import FrequencyResponse
from .config import BONE_CONDUCTION_KEYWORDS

def is_bone_conduction_device(name: str) -> bool:
    """Checks if a headphone model name indicates bone conduction."""
    if not name:
        return False
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in BONE_CONDUCTION_KEYWORDS)

def github_to_raw_url(html_url: str) -> str:
    """Converts a standard GitHub file URL to a raw content URL."""
    return html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

def search_autoeq_files(headphone_name: str) -> list[dict]:
    """Searches the GitHub Code Search API for a headphone model name in AutoEq results.
    
    Returns a list of dicts with keys: 'name', 'path', 'database', 'html_url', 'raw_url'
    """
    if not headphone_name:
        return []
    
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
                    "name": item.get("name", "").replace(".csv", ""),
                    "path": path,
                    "database": db_source,
                    "html_url": item.get("html_url", ""),
                    "raw_url": github_to_raw_url(item.get("html_url", ""))
                })
            return results
            
    except Exception as e:
        print(f"GitHub Search API call failed: {e}.")
        return []

def fetch_frequency_response(raw_url: str, headphone_name: str = "") -> FrequencyResponse:
    """Retrieves and parses the CSV response data from a raw AutoEq URL.
    
    Returns a FrequencyResponse object.
    """
    headers = {
        "User-Agent": "HearingTestCalibrationAgent/1.0"
    }
    
    try:
        req = urllib.request.Request(raw_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode('utf-8')
            
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        try:
            # FrequencyResponse in autoeq 4.x uses read_csv, while autoeq 2.2.0 uses read_from_csv
            if hasattr(FrequencyResponse, 'read_csv'):
                fr = FrequencyResponse.read_csv(temp_file_path)
            else:
                fr = FrequencyResponse.read_from_csv(temp_file_path)
            fr.name = headphone_name
            return fr
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        print(f"Failed to fetch CSV data from {raw_url}: {e}")
        raise e
