import shutil, psutil, os, sys

# Disk space
total, used, free = shutil.disk_usage(".")
print("=== DISK SPACE ===")
print("Drive:", os.path.abspath(".")[:3])
print("Total:", round(total / (1024**3), 2), "GB")
print("Used :", round(used / (1024**3), 2), "GB")
print("Free :", round(free / (1024**3), 2), "GB")

# RAM & CPU
print("\n=== SYSTEM HARDWARE ===")
mem = psutil.virtual_memory()
print("Total RAM    :", round(mem.total / (1024**3), 2), "GB")
print("Available RAM:", round(mem.available / (1024**3), 2), "GB")
print("RAM Used %   :", mem.percent, "%")
print("Logical CPUs :", psutil.cpu_count(logical=True))
print("Physical CPUs:", psutil.cpu_count(logical=False))

# Python & ML packages
print("\n=== PYTHON & PACKAGES ===")
print("Python Version:", sys.version.split()[0])
for pkg in ["sklearn", "scipy", "numpy", "pandas", "joblib"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "available")
        print(f"  {pkg:12}: {ver}")
    except ImportError:
        print(f"  {pkg:12}: NOT INSTALLED")
