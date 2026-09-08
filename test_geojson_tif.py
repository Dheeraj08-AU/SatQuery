import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from app import generate_gis_geojson

print("Testing GeoTIFF extraction...")
tif_path = "sanity_imgs/RGB.byte.tif"

if not os.path.exists(tif_path):
    print(f"File not found: {tif_path}")
    sys.exit(1)

geojson = generate_gis_geojson(tif_path, "Test Task", "Test Description", 0.99)

print("\n--- GeoJSON Output ---")
print(json.dumps(geojson, indent=2))
