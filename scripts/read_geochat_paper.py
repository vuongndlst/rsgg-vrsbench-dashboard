import urllib.request
import re

url = 'https://arxiv.org/html/2311.15826v2' # or v1
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    print(f"Fetched HTML, length: {len(html)} bytes")
    # Search for headings and tables
    sections = re.findall(r'<h[234][^>]*>(.*?)</h[234]>', html)
    print("SECTIONS:")
    for s in sections[:30]:
        print(" -", re.sub(r'<[^>]+>', '', s).strip())

    # Search for tables
    tables = re.findall(r'<caption[^>]*>(.*?)</caption>', html, re.DOTALL)
    print("\nTABLE CAPTIONS:")
    for t in tables:
        print(" *", re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', t)).strip())
except Exception as e:
    print("HTML failed:", e)
    # Fallback to abs
    url2 = 'https://arxiv.org/abs/2311.15826'
    req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    html2 = urllib.request.urlopen(req2, timeout=10).read().decode('utf-8')
    print("Abs fetched:", len(html2))

