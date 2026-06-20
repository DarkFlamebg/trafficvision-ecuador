import urllib.request
import re
try:
    html = urllib.request.urlopen('https://download.openmmlab.com/mmcv/dist/cu121/index.html').read().decode()
    matches = re.findall(r'href="(torch[^/]+)/"', html)
    print(set(matches))
except Exception as e:
    print('error', e)
