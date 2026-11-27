#!/usr/bin/env python3
"""Download and validate CWRU control file 109.mat (48 kHz Drive-End fault).

This file is not part of the 40-file dataset and is used exclusively as a
diagnostic control to evaluate sampling-rate and filter roll-off behavior.
"""
# Reads: existing data/raw/109.mat (via scipy.io.loadmat) or remote CWRU URL
# Writes: data/raw/109.mat (staged via data/raw/109.part)

from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import scipy.io


def validate_file(path: Path) -> bool:
    """Validate 109.mat size, structure, and sample count."""
    if not path.is_file():
        return False
    # Reject truncated downloads at or below 100 KiB (102400 bytes)
    if path.stat().st_size <= 100 * 1024:
        return False
    try:
        mat = scipy.io.loadmat(str(path))
        if "X109_DE_time" not in mat:
            return False
        data = mat["X109_DE_time"].squeeze()
        # Require exact sample count (243938 samples) of 48 kHz recording 109.mat
        if len(data) != 243938:
            return False
        return True
    except Exception:
        return False


def download_and_validate(dest_path: Path | str, max_retries: int = 3, timeout: int = 30) -> bool:
    """Download 109.mat with retries and validate contents."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    url = "https://engineering.case.edu/sites/default/files/109.mat"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    downloaded = False
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                part.write_bytes(content)
            downloaded = True
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries:
                # Linear backoff: wait 1.5 * attempt seconds before retry (max_retries=3, timeout=30 s)
                time.sleep(1.5 * attempt)
            else:
                if part.exists():
                    part.unlink()
                print(f"[ERROR] Failed to download 109.mat after {max_retries} attempts: {e}")
                return False

    if not downloaded:
        if part.exists():
            part.unlink()
        return False

    if not validate_file(part):
        if part.exists():
            part.unlink()
        print(f"[ERROR] Validation failed for {part}; file removed.")
        return False

    part.replace(dest)
    return True


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    target = root / "data" / "raw" / "109.mat"

    if target.exists():
        if validate_file(target):
            print("already present and valid")
            return
        else:
            print("File exists but is invalid; delete it and run the script again.")
            sys.exit(1)

    success = download_and_validate(target)
    if not success:
        if target.exists():
            target.unlink()
        print(f"[ERROR] Failed to download and validate {target}.")
        sys.exit(1)
    print(f"Downloaded and validated {target} successfully.")


if __name__ == "__main__":
    main()
