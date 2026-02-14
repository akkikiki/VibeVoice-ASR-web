"""Deploy the built dist/ to a HuggingFace Space (static site)."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

SPACE_ID = "akkikiki/VibeVoice-ASR"
DIST_DIR = Path(__file__).parent / "dist"


def collect_files(dist_dir: Path) -> list[tuple[str, Path]]:
    """Collect all files in dist/ with their repo-relative paths."""
    files = []
    for path in sorted(dist_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(dist_dir)
            files.append((str(rel), path))
    return files


def main():
    if not DIST_DIR.exists():
        print("Error: dist/ not found. Run 'npm run build' first.")
        sys.exit(1)

    api = HfApi()

    # Check if Space exists, create if not
    try:
        api.repo_info(repo_id=SPACE_ID, repo_type="space")
        print(f"Space {SPACE_ID} exists")
    except Exception:
        print(f"Creating Space {SPACE_ID}...")
        api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="static")

    # Collect dist files
    files = collect_files(DIST_DIR)
    print(f"Found {len(files)} files in dist/")

    # Build operations: upload all dist files + Space README
    operations = []

    # Space README with metadata (must be at root)
    readme_content = """---
title: VibeVoice ASR
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: mit
---
"""
    operations.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=readme_content.encode("utf-8"),
    ))

    for rel_path, abs_path in files:
        print(f"  Adding: {rel_path} ({abs_path.stat().st_size:,} bytes)")
        operations.append(CommitOperationAdd(
            path_in_repo=rel_path,
            path_or_fileobj=str(abs_path),
        ))

    # Delete old files that may no longer exist
    try:
        existing = api.list_repo_files(repo_id=SPACE_ID, repo_type="space")
        new_paths = {"README.md"} | {rel for rel, _ in files}
        for old in existing:
            if old not in new_paths and old != ".gitattributes":
                print(f"  Deleting old: {old}")
                operations.append(CommitOperationDelete(path_in_repo=old))
    except Exception:
        pass

    print(f"\nUploading {len(operations)} operations...")
    api.create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=operations,
        commit_message="Deploy VibeVoice ASR web app with model selector",
    )
    print(f"\nDone! Space: https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
