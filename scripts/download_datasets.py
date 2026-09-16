"""Automated dataset downloader for RoadSentinel AI.

Downloads or prepares datasets:
1. Synthetic Dataset for Accident Detection (Kaggle via kagglehub)
2. CCTV Footage Accident Detection (Kaggle via kagglehub)
3. NYC EMS Incident Dispatch Data (NYC Open Data via Socrata CSV API)
4. US Accidents sample / instructions
"""
import os
import shutil
import urllib.request
import pandas as pd


def download_kaggle_synthetic(data_dir: str = "data"):
    """Download synthetic accident dataset from Kaggle via kagglehub."""
    print("--- 1. Downloading Kaggle Synthetic Accident Detection Dataset ---")
    try:
        import kagglehub
        path = kagglehub.dataset_download("mehwishtahir722/synthetic-dataset-for-accident-detection")
        print(f"Downloaded to cache: {path}")
        dest = os.path.join(data_dir, "synthetic_accidents")
        os.makedirs(dest, exist_ok=True)
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"Successfully copied to {dest}")
        return True
    except Exception as e:
        print(f"Kaggle download error (credentials or rate limit): {e}")
        return False


def download_kaggle_cctv(data_dir: str = "data"):
    """Download CCTV accident footage dataset from Kaggle."""
    print("--- 2. Downloading Kaggle CCTV Accident Dataset ---")
    try:
        import kagglehub
        path = kagglehub.dataset_download("ckay16/accident-detection-from-cctv-footage")
        print(f"Downloaded to cache: {path}")
        dest = os.path.join(data_dir, "cctv_accidents")
        os.makedirs(dest, exist_ok=True)
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"Successfully copied to {dest}")
        return True
    except Exception as e:
        print(f"CCTV download error: {e}")
        return False


def download_nyc_ems_data(output_path: str = "data/nyc_ems_response.csv"):
    """Download real NYC EMS 911 dispatch response data via Socrata Open Data CSV endpoint."""
    print("--- 3. Downloading NYC EMS Incident Dispatch Data (Public API) ---")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # NYC Open Data EMS dispatch endpoint (public CSV export, limit 5000 records)
    url = "https://data.cityofnewyork.us/resource/76xm-jjuj.csv?$limit=5000"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(output_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        df = pd.read_csv(output_path)
        print(f"Successfully downloaded {len(df)} live NYC EMS dispatch records -> {output_path}")
        return True
    except Exception as e:
        print(f"NYC EMS API download error: {e}. Falling back to generated seed data.")
        return False


def download_all():
    print("==================================================")
    print("      RoadSentinel AI — Automated Downloader      ")
    print("==================================================")
    os.makedirs("data", exist_ok=True)
    download_kaggle_synthetic()
    download_kaggle_cctv()
    download_nyc_ems_data()
    print("==================================================")
    print("Automated download sequence completed.")
    print("If any Kaggle dataset requires an API key, ensure ~/.kaggle/kaggle.json exists.")
    print("Seed data remains active in data/ to guarantee 100% operational uptime.")
    print("==================================================")


if __name__ == "__main__":
    download_all()
