import urllib.request
import re

url = "https://arxiv.org/html/2406.12384v1"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, timeout=20).read().decode('utf-8', errors='ignore')
    # Find tables
    tables = re.findall(r'(<table.*?</table>)', html, re.DOTALL)
    print(f"Found {len(tables)} tables")
    for i, t in enumerate(tables):
        if "Detailed image caption" in t or "GeoChat" in t or "LLaVA" in t:
            # strip tags to text
            clean = re.sub(r'<[^>]+>', '\t', t)
            clean = re.sub(r'\t+', '\t', clean)
            clean = re.sub(r'\n+', '\n', clean)
            print(f"\n--- TABLE {i+1} ---")
            print(clean[:1500])
except Exception as e:
    print("Fetch error:", e)

