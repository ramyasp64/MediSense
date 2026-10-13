"""
Download all MediSense datasets.

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --bidmc --capnobase --nasa
    python scripts/download_data.py --mimic   # only after PhysioNet access granted
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data/raw")


def run(cmd: str, cwd=None):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def download_bidmc():
    out = DATA_DIR / "bidmc"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[1/5] Downloading PhysioNet BIDMC PPG dataset...")
    print("  Using wfdb Python library (no manual download needed)")
    script = f"""
import wfdb
wfdb.dl_database('bidmc', '{out}')
print('BIDMC download complete.')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def download_capnobase():
    out = DATA_DIR / "capnobase"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[2/5] CapnoBase dataset")
    print("  Manual download required (Borealis Data repository).")
    print("  1. Go to: https://borealisdata.ca/dataverse/capnobase")
    print("  2. Download the .zip file")
    print(f"  3. Extract .mat files into: {out.absolute()}/")
    print("  (We'll proceed — place files there before running experiments)")


def download_nasa_bearing():
    out = DATA_DIR / "nasa_bearing"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[3/5] NASA Bearing dataset")
    print("  Manual download required from NASA Prognostics Data Repository.")
    print("  1. Go to: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print("  2. Download 'Bearing Data Set' (IMS)")
    print(f"  3. Extract 1st_test/, 2nd_test/, 3rd_test/ into: {out.absolute()}/")


def download_nasa_cmapss():
    out = DATA_DIR / "nasa_cmapss"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[4/5] NASA CMAPSS Turbofan dataset")
    print("  Attempting direct download from NASA...")
    try:
        run(
            "wget -q -O CMAPSSData.zip "
            "'https://ti.arc.nasa.gov/c/6/' "
            f"&& unzip -q CMAPSSData.zip -d '{out}' && rm CMAPSSData.zip"
        )
        print("  CMAPSS download complete.")
    except Exception:
        print("  Direct download failed. Manual steps:")
        print("  1. Go to: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print("  2. Download 'Turbofan Engine Degradation Simulation'")
        print(f"  3. Extract train_FD001.txt, RUL_FD001.txt etc. into: {out.absolute()}/")


def download_skab():
    out = DATA_DIR / "skab"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[5/5] SKAB (Skoltech Anomaly Benchmark)...")
    try:
        run(f"git clone --depth=1 https://github.com/waico/SKAB.git /tmp/skab_tmp")
        run(f"cp -r /tmp/skab_tmp/data/* '{out}/'")
        run("rm -rf /tmp/skab_tmp")
        print("  SKAB download complete.")
    except Exception:
        print("  git clone failed. Manual steps:")
        print("  1. Download: https://github.com/waico/SKAB/archive/refs/heads/main.zip")
        print(f"  2. Extract data/ folder contents into: {out.absolute()}/")


def download_mimic():
    out = DATA_DIR / "mimic"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[MIMIC-III] Downloading matched waveform subset...")
    print("  Requires PhysioNet credentialing. Ensure you have completed CITI training")
    print("  and have been granted access at physionet.org before running this.")
    script = f"""
import wfdb
# Download a small matched subset first to verify access
wfdb.dl_database('mimic3wdb-matched', '{out}', records=['p00/p000020/3141595'])
print('MIMIC-III sample download complete. Expand records list for full dataset.')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download MediSense datasets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bidmc", action="store_true")
    parser.add_argument("--capnobase", action="store_true")
    parser.add_argument("--nasa", action="store_true")
    parser.add_argument("--mimic", action="store_true")
    args = parser.parse_args()

    if args.all or args.bidmc:
        download_bidmc()
    if args.all or args.capnobase:
        download_capnobase()
    if args.all or args.nasa:
        download_nasa_bearing()
        download_nasa_cmapss()
        download_skab()
    if args.mimic:
        download_mimic()

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
