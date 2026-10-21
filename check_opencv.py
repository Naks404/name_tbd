import urllib.request
import re
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request('https://chaquo.com/pypi-7.0/', headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, context=ctx).read().decode('utf-8')
    matches = [m.group(1) for m in re.finditer(r'href=[\"\']([a-zA-Z0-9-]+)/?[\"\']', html) if 'opencv' in m.group(1).lower()]
    print("MATCHES:", matches)
except Exception as e:
    print("ERROR:", e)