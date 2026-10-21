"""Fetch publisher-sharded Qwen 7B GGUF; llama.cpp automatically opens both shards."""

import json
import urllib.request
from pathlib import Path

from edge_agent_lab.common import sha256, write_json

root = Path("data/models/qwen7b")
root.mkdir(parents=True, exist_ok=True)
repo = "Qwen/Qwen2.5-7B-Instruct-GGUF"
with urllib.request.urlopen(
    f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60
) as response:
    meta = json.load(response)
files = [row for row in meta["siblings"] if "q4_k_m-" in row["rfilename"]]
manifest = {"repository": repo, "revision": meta["sha"], "license": "Apache-2.0", "files": []}
for row in files:
    name, digest = row["rfilename"], row["lfs"]["sha256"]
    path = root / name
    if not path.exists():
        partial = path.with_suffix(".part")
        url = f"https://huggingface.co/{repo}/resolve/{meta['sha']}/{name}"
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if sha256(partial) != digest:
            raise ValueError(f"Checksum failed: {name}")
        partial.replace(path)
    if sha256(path) != digest:
        raise ValueError(f"Cached checksum failed: {name}")
    manifest["files"].append({"filename": name, "sha256": digest, "bytes": path.stat().st_size})
    write_json(root / "manifest.json", manifest)
print("All shards verified", flush=True)
