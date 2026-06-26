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

_AUTOEQ_INDEX_CACHE = None

def search_autoeq_files(headphone_name: str) -> list[dict]:
    """Retrieves all measurement entries for a headphone model name from the AutoEq index.
    
    Returns a list of dicts with keys: 'name', 'path', 'database', 'html_url', 'raw_url'
    """
    global _AUTOEQ_INDEX_CACHE
    if not headphone_name:
        return []
        
    if _AUTOEQ_INDEX_CACHE is None:
        url = "https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/INDEX.md"
        headers = {
            "User-Agent": "HearingTestCalibrationAgent/1.0"
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8')
                
            # Regex to parse the markdown index lines
            pattern = re.compile(
                r'^-\s+\[(?P<name>[^\]]+)\]\(\./(?P<path>[^\)]+)\)\s+by\s+(?P<source>.*?)(?:\s+on\s+(?P<rig>[^\n]+))?$',
                re.MULTILINE
            )
            
            entries = []
            for match in pattern.finditer(content):
                gd = match.groupdict()
                entries.append({
                    "name": gd["name"],
                    "path": gd["path"],
                    "source": gd["source"],
                    "rig": gd["rig"] if gd["rig"] else ""
                })
            _AUTOEQ_INDEX_CACHE = entries
        except Exception as e:
            print(f"Failed to fetch or parse AutoEq index: {e}")
            return []
            
    # Filter entries matching headphone_name case-insensitively
    matches = [e for e in _AUTOEQ_INDEX_CACHE if e["name"].lower() == headphone_name.lower()]
    
    results = []
    for entry in matches:
        path = entry["path"]
        last_part = path.split('/')[-1]
        
        # database is the first part of the path (e.g. 'oratory1990', 'crinacle', 'Rtings')
        path_parts = path.split("/")
        db_source = path_parts[0] if path_parts else "unknown"
        
        # Build html_url and raw_url
        html_url = f"https://github.com/jaakkopasanen/AutoEq/blob/master/results/{path}/{last_part}.csv"
        raw_url = f"https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/{path}/{last_part}.csv"
        
        results.append({
            "name": entry["name"],
            "path": path,
            "database": db_source,
            "html_url": html_url,
            "raw_url": raw_url
        })
        
    return results

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
