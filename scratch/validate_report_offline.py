import re
from pathlib import Path

html_path = Path("reports/cluster_visualization_v1/cluster_inspection_report.html")
html = html_path.read_text(encoding="utf-8")

print("Checking external dependencies in HTML:")
script_src = re.findall(r'<script[^>]+src=[^>]+>', html, re.IGNORECASE)
link_css = re.findall(r'<link[^>]+rel=[\'"]?stylesheet[\'"]?[^>]+>', html, re.IGNORECASE)
font_imports = re.findall(r'@import\s+[^;]+;', html, re.IGNORECASE)
font_urls = re.findall(r'url\s*\(\s*[\'"]?https?://[^)]+\)', html, re.IGNORECASE)

print(f"  <script src=...>: {script_src}")
print(f"  <link rel='stylesheet'...>: {link_css}")
print(f"  @import font rules: {font_imports}")
print(f"  CSS url(http...): {font_urls}")

is_offline = (len(script_src) == 0 and len(link_css) == 0 and len(font_imports) == 0 and len(font_urls) == 0)
print(f"\n100% Offline Compatible: {is_offline}")
