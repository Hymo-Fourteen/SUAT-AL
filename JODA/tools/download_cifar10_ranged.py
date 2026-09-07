import concurrent.futures
import hashlib
import os
import sys
import time
from pathlib import Path

import requests

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
TARGET = Path("data/cifar10/cifar-10-python.tar.gz")
PARTS = TARGET.parent / ".parts"
TOTAL = 170498071
CHUNK = 4 * 1024 * 1024
WORKERS = 6
MD5 = "c58f30108f718f92721af3b95e74349a"


def chunk_bounds(i):
    return i * CHUNK, min((i + 1) * CHUNK, TOTAL)


def part_path(i):
    return PARTS / f"part_{i:03d}.bin"


def fetch(i):
    start, end = chunk_bounds(i)
    length = end - start
    path = part_path(i)
    have = path.stat().st_size if path.exists() else 0
    if have >= length:
        return i, length, True
    headers = {"Range": f"bytes={start + have}-{end - 1}"}
    last_err = None
    for attempt in range(4):
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(path, "ab") as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            have += len(chunk)
            return i, length, have >= length
        except Exception as e:
            last_err = e
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"part {i} failed after retries: {last_err}")


def progress():
    total = sum(p.stat().st_size for p in PARTS.glob("part_*.bin"))
    return total, TOTAL


def merge():
    print("Merging parts...", flush=True)
    with open(TARGET, "wb") as out:
        for i in range((TOTAL + CHUNK - 1) // CHUNK):
            with open(part_path(i), "rb") as f:
                out.write(f.read())
    h = hashlib.md5()
    with open(TARGET, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    print("md5:", h.hexdigest(), "expected:", MD5, flush=True)


def main():
    max_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 450.0
    PARTS.mkdir(parents=True, exist_ok=True)
    n = (TOTAL + CHUNK - 1) // CHUNK
    todo = [i for i in range(n) if not part_path(i).exists() or part_path(i).stat().st_size < (chunk_bounds(i)[1] - chunk_bounds(i)[0])]
    print(f"start, {len(todo)} parts pending", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs = {ex.submit(fetch, i): i for i in todo}
        deadline = time.time() + max_seconds
        done = 0
        try:
            for fut in concurrent.futures.as_completed(fs, timeout=max_seconds):
                i, length, ok = fut.result()
                done += 1
                have, total = progress()
                print(f"[{time.time()-deadline+max_seconds:6.1f}s left] part {i} ok={ok} done={done}/{len(todo)} bytes={have}/{total} ({100.0*have/total:.1f}%)", flush=True)
                if time.time() >= deadline:
                    break
        except concurrent.futures.TimeoutError:
            pass
        for fut in fs:
            fut.cancel()
    have, total = progress()
    print(f"stopping with {have}/{total} bytes ({100.0*have/total:.1f}%)", flush=True)
    if have >= total:
        merge()
        print("download complete", flush=True)


if __name__ == "__main__":
    main()
