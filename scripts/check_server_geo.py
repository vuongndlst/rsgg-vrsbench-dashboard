import urllib.request, json
try:
    with urllib.request.urlopen("https://ipinfo.io/json", timeout=5) as r:
        print(r.read().decode())
except Exception as e:
    print("Error:", e)

