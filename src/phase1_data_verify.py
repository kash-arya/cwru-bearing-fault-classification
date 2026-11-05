"""Phase 1: Data Acquisition and Verification Pipeline for CWRU Bearing Dataset.

Implements Steps 1-4 of roadmap.md:
- Strictly validates and acquires 40 CWRU drive-end MAT files (excluding 0.028").
- Validates file size (>100KB), loadmat readability, and exact DE_time key match.
- Detects extraneous DE keys (e.g. workspace leakage in 99.mat) without halting.
- Strict RPM resolution: exact key f"X{file_id:03d}RPM" or deterministic load fallback.
- Resamples 48 kHz Normal baseline signals to 12 kHz using polyphase FIR filtering.
- Performs window counting (2048 window, 50% overlap) and anomaly flag logging.
- Generates results/phase1_file_manifest.csv and caches data/processed/phase1_signals_12k.pkl.
"""

import csv
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import scipy.io
import scipy.signal

# -----------------------------------------------------------------------------
# Configuration: Explicit 40-ID CWRU Dataset Specification
# -----------------------------------------------------------------------------
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

FILES_CONFIG = [
    # Normal Baseline (48 kHz)
    {"id": 97, "class": "Normal", "severity": "none", "load_hp": 0},
    {"id": 98, "class": "Normal", "severity": "none", "load_hp": 1},
    {"id": 99, "class": "Normal", "severity": "none", "load_hp": 2},
    {"id": 100, "class": "Normal", "severity": "none", "load_hp": 3},
    # Inner Race Fault (12 kHz) - 0.007"
    {"id": 105, "class": "Inner Race", "severity": "0.007", "load_hp": 0},
    {"id": 106, "class": "Inner Race", "severity": "0.007", "load_hp": 1},
    {"id": 107, "class": "Inner Race", "severity": "0.007", "load_hp": 2},
    {"id": 108, "class": "Inner Race", "severity": "0.007", "load_hp": 3},
    # Inner Race Fault (12 kHz) - 0.014"
    {"id": 169, "class": "Inner Race", "severity": "0.014", "load_hp": 0},
    {"id": 170, "class": "Inner Race", "severity": "0.014", "load_hp": 1},
    {"id": 171, "class": "Inner Race", "severity": "0.014", "load_hp": 2},
    {"id": 172, "class": "Inner Race", "severity": "0.014", "load_hp": 3},
    # Inner Race Fault (12 kHz) - 0.021"
    {"id": 209, "class": "Inner Race", "severity": "0.021", "load_hp": 0},
    {"id": 210, "class": "Inner Race", "severity": "0.021", "load_hp": 1},
    {"id": 211, "class": "Inner Race", "severity": "0.021", "load_hp": 2},
    {"id": 212, "class": "Inner Race", "severity": "0.021", "load_hp": 3},
    # Ball Fault (12 kHz) - 0.007"
    {"id": 118, "class": "Ball", "severity": "0.007", "load_hp": 0},
    {"id": 119, "class": "Ball", "severity": "0.007", "load_hp": 1},
    {"id": 120, "class": "Ball", "severity": "0.007", "load_hp": 2},
    {"id": 121, "class": "Ball", "severity": "0.007", "load_hp": 3},
    # Ball Fault (12 kHz) - 0.014"
    {"id": 185, "class": "Ball", "severity": "0.014", "load_hp": 0},
    {"id": 186, "class": "Ball", "severity": "0.014", "load_hp": 1},
    {"id": 187, "class": "Ball", "severity": "0.014", "load_hp": 2},
    {"id": 188, "class": "Ball", "severity": "0.014", "load_hp": 3},
    # Ball Fault (12 kHz) - 0.021"
    {"id": 222, "class": "Ball", "severity": "0.021", "load_hp": 0},
    {"id": 223, "class": "Ball", "severity": "0.021", "load_hp": 1},
    {"id": 224, "class": "Ball", "severity": "0.021", "load_hp": 2},
    {"id": 225, "class": "Ball", "severity": "0.021", "load_hp": 3},
    # Outer Race Fault @ 6:00 (12 kHz) - 0.007"
    {"id": 130, "class": "Outer Race", "severity": "0.007", "load_hp": 0},
    {"id": 131, "class": "Outer Race", "severity": "0.007", "load_hp": 1},
    {"id": 132, "class": "Outer Race", "severity": "0.007", "load_hp": 2},
    {"id": 133, "class": "Outer Race", "severity": "0.007", "load_hp": 3},
    # Outer Race Fault @ 6:00 (12 kHz) - 0.014"
    {"id": 197, "class": "Outer Race", "severity": "0.014", "load_hp": 0},
    {"id": 198, "class": "Outer Race", "severity": "0.014", "load_hp": 1},
    {"id": 199, "class": "Outer Race", "severity": "0.014", "load_hp": 2},
    {"id": 200, "class": "Outer Race", "severity": "0.014", "load_hp": 3},
    # Outer Race Fault @ 6:00 (12 kHz) - 0.021"
    {"id": 234, "class": "Outer Race", "severity": "0.021", "load_hp": 0},
    {"id": 235, "class": "Outer Race", "severity": "0.021", "load_hp": 1},
    {"id": 236, "class": "Outer Race", "severity": "0.021", "load_hp": 2},
    {"id": 237, "class": "Outer Race", "severity": "0.021", "load_hp": 3},
]


def download_file(file_id: int, dest_path: Path, max_retries: int = 3, timeout: int = 30) -> bool:
    """Download .mat file from official CWRU server with retries."""
    url = f"https://engineering.case.edu/sites/default/files/{file_id}.mat"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                dest_path.write_bytes(content)
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
            else:
                print(f"[ERROR] Failed to download file {file_id}.mat after {max_retries} attempts: {e}")
                return False
    return False


def validate_mat_file(file_path: Path, file_id: int) -> tuple[bool, str, dict, list[str]]:
    """Validate file size, readable MAT format, and presence of exact expected DE_time key."""
    if not file_path.exists():
        return False, f"File {file_path.name} does not exist.", {}, []

    size_bytes = file_path.stat().st_size
    if size_bytes <= 100 * 1024:
        return False, f"File {file_path.name} size {size_bytes} <= 100 KB threshold.", {}, []

    try:
        mat_data = scipy.io.loadmat(str(file_path))
    except Exception as e:
        return False, f"Failed to load MAT file {file_path.name}: {e}", {}, []

    expected_de_key = f"X{file_id:03d}_DE_time"
    if expected_de_key not in mat_data:
        de_keys = [k for k in mat_data.keys() if k.endswith("DE_time")]
        return (
            False,
            f"Missing required key '{expected_de_key}'. Available DE keys: {de_keys}",
            {},
            [],
        )

    other_de_keys = sorted([k for k in mat_data.keys() if k.endswith("DE_time") and k != expected_de_key])
    return True, "", mat_data, other_de_keys


def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "raw"
    proc_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    verif_dir = root_dir / "verification"
    tmp_cache_dir = Path("/tmp/cwru_test")

    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Populate raw directory from available local caches before initiating web downloads
    for cfg in FILES_CONFIG:
        fid = cfg["id"]
        fname = f"{fid}.mat"
        dest_file = raw_dir / fname
        if not dest_file.exists():
            candidate_paths = [root_dir / fname, verif_dir / fname, tmp_cache_dir / fname]
            for cand in candidate_paths:
                if cand.exists():
                    shutil.copy2(cand, dest_file)
                    break

    # Acquire and validate all 40 files
    missing_ids = []
    validated_data = {}
    extra_de_keys_map = {}
    downloaded_this_run = []

    for cfg in FILES_CONFIG:
        fid = cfg["id"]
        fname = f"{fid}.mat"
        target_path = raw_dir / fname

        if not target_path.exists():
            success = download_file(fid, target_path)
            if success:
                downloaded_this_run.append(target_path)
            else:
                missing_ids.append(fid)
                continue

        # Validate file integrity with exact key matching
        is_valid, err_msg, mat_dict, extra_keys = validate_mat_file(target_path, fid)
        if not is_valid:
            print(f"[VALIDATION FAILED] {err_msg}")
            if target_path in downloaded_this_run and target_path.exists():
                target_path.unlink()
            sys.exit(1)

        validated_data[fid] = mat_dict
        extra_de_keys_map[fid] = extra_keys

    if missing_ids:
        print(f"[STOP] Missing files that could not be downloaded: {missing_ids}")
        print("Please download these files manually into data/raw/.")
        sys.exit(1)

    # Process all signals
    manifest_rows = []
    signals_dict = {}

    for cfg in FILES_CONFIG:
        fid = cfg["id"]
        fclass = cfg["class"]
        severity = cfg["severity"]
        load_hp = cfg["load_hp"]
        fname = f"{fid}.mat"
        de_key = f"X{fid:03d}_DE_time"
        other_keys = extra_de_keys_map[fid]
        other_de_keys_str = ";".join(other_keys) if other_keys else ""

        mat_data = validated_data[fid]
        raw_x = mat_data[de_key].squeeze().astype(np.float64)
        raw_samples = len(raw_x)

        # Strict RPM extraction: exact key f"X{fid:03d}RPM" only
        expected_rpm_key = f"X{fid:03d}RPM"
        if expected_rpm_key in mat_data:
            rpm = float(mat_data[expected_rpm_key].squeeze())
            rpm_source = "mat_file"
        else:
            rpm = float(NOMINAL_RPM[load_hp])
            rpm_source = "fallback"

        # Signal resampling logic
        if fclass == "Normal":
            if raw_samples <= 200000:
                print(f"[STOP] File {fname} normal length {raw_samples} <= 200,000 samples. Unexpected length.")
                sys.exit(1)
            x_12k = scipy.signal.resample_poly(raw_x, 1, 4)
        else:
            x_12k = raw_x

        resampled_12k_samples = len(x_12k)
        window_count = (resampled_12k_samples - 2048) // 1024 + 1

        # Anomaly flags
        flags = []
        if other_keys:
            flags.append("multiple_de_keys")
        if fclass != "Normal":
            if abs(raw_samples - 121000) / 121000 > 0.05:
                flags.append("length_deviates")
        if rpm_source == "mat_file":
            if abs(rpm - NOMINAL_RPM[load_hp]) > 20:
                flags.append("rpm_load_mismatch")
        elif rpm_source == "fallback":
            flags.append("rpm_fallback")

        anomaly_flag = ";".join(flags)

        row = {
            "file_id": fid,
            "filename": fname,
            "fault_class": fclass,
            "severity": severity,
            "load_hp": load_hp,
            "raw_samples": raw_samples,
            "resampled_12k_samples": resampled_12k_samples,
            "rpm": rpm,
            "rpm_source": rpm_source,
            "window_count": window_count,
            "anomaly_flag": anomaly_flag,
            "de_key": de_key,
            "other_de_keys": other_de_keys_str,
        }
        manifest_rows.append(row)
        signals_dict[fid] = {"x": x_12k, "meta": row}

    # Save outputs
    manifest_path = results_dir / "phase1_file_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_id",
            "filename",
            "fault_class",
            "severity",
            "load_hp",
            "raw_samples",
            "resampled_12k_samples",
            "rpm",
            "rpm_source",
            "window_count",
            "anomaly_flag",
            "de_key",
            "other_de_keys",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    pkl_path = proc_dir / "phase1_signals_12k.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(signals_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print summary tables to terminal
    print("\n" + "=" * 135)
    print(
        f"{'ID':>4} | {'Filename':>8} | {'Class':<11} | {'Sev':<6} | {'HP':>2} | {'Raw Samp':>9} | {'12k Samp':>9} | {'RPM':>6} | {'RPM Src':<8} | {'Windows':>7} | {'Flags':<22} | {'Other DE Keys'}"
    )
    print("-" * 135)
    non_empty_flags_count = 0
    for r in manifest_rows:
        flag_str = r["anomaly_flag"] if r["anomaly_flag"] else "-"
        other_str = r["other_de_keys"] if r["other_de_keys"] else "-"
        if r["anomaly_flag"]:
            non_empty_flags_count += 1
        print(
            f"{r['file_id']:4d} | {r['filename']:>8} | {r['fault_class']:<11} | {r['severity']:<6} | {r['load_hp']:2d} | "
            f"{r['raw_samples']:9d} | {r['resampled_12k_samples']:9d} | {r['rpm']:6.1f} | {r['rpm_source']:<8} | {r['window_count']:7d} | {flag_str:<22} | {other_str}"
        )
    print("=" * 135)

    # Summary per class
    print("\n--- Summary Per Class ---")
    class_summary = {}
    for r in manifest_rows:
        cls = r["fault_class"]
        if cls not in class_summary:
            class_summary[cls] = {"files": 0, "windows": 0}
        class_summary[cls]["files"] += 1
        class_summary[cls]["windows"] += r["window_count"]

    print(f"{'Class':<12} | {'File Count':>10} | {'Total Windows':>13}")
    print("-" * 42)
    total_files = 0
    total_windows = 0
    for cls, stats in class_summary.items():
        total_files += stats["files"]
        total_windows += stats["windows"]
        print(f"{cls:<12} | {stats['files']:10d} | {stats['windows']:13d}")
    print("-" * 42)
    print(f"{'TOTAL':<12} | {total_files:10d} | {total_windows:13d}")

    print(f"\nFiles with non-empty anomaly flag: {non_empty_flags_count}")

    # Explicit audit print for 99.mat (Normal_2)
    row_99 = next(r for r in manifest_rows if r["file_id"] == 99)
    print("\n--- File 99.mat Audit ---")
    print(f"  Exact Key Used   : {row_99['de_key']}")
    print(f"  Raw Length       : {row_99['raw_samples']} samples (48 kHz)")
    print(f"  Resampled Length : {row_99['resampled_12k_samples']} samples (12 kHz)")
    print(f"  RPM Used         : {row_99['rpm']} ({row_99['rpm_source']})")
    print(f"  Other DE Keys    : {row_99['other_de_keys']}")
    print(f"  Anomaly Flags    : {row_99['anomaly_flag']}")

    if total_files == 40:
        print("\nPHASE 1 PASS: 40/40")


if __name__ == "__main__":
    main()
