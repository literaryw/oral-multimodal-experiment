"""
Download Lab-Rasool/TCGA features with progress bar
"""
import os, certifi, sys
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("../data/TCGA-HNSC/pretrained_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/Lab-Rasool/TCGA/resolve/main"

FILES = {
    "Slide_Image_UNI.parquet": f"{BASE_URL}/Slide%20Image%20(UNI)/Slide%20Image%20(UNI).parquet",
    "Clinical_GatorTron.parquet": f"{BASE_URL}/Clinical%20Data%20(gatortron-base)/data-00000-of-00001.parquet",
    "Pathology_Report_GatorTron.parquet": f"{BASE_URL}/Pathology%20Report%20(gatortron-base)/data-00000-of-00001.parquet",
}

for name, url in FILES.items():
    out_path = OUTPUT_DIR / name
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"[OK] {name} exists ({size_mb:.0f} MB), skip")
        continue

    print(f"\n[DL] {name}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    print(f"     Size: {total / 1e6:.0f} MB")

    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1e6
                total_mb = total / 1e6
                print(f"\r     {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="")
                sys.stdout.flush()

    print(f"\n     Done ({out_path.stat().st_size / 1e6:.0f} MB)")

print("\n=== Verify ===")
for f in sorted(OUTPUT_DIR.glob("*.parquet")):
    try:
        df = pd.read_parquet(f)
        print(f"  {f.name}: {df.shape}")
    except Exception as e:
        print(f"  {f.name}: ERROR - {e}")
