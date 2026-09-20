# GitHub Upload Guide — Source and Revised Study 1 Assets

This archive is a repository source package with the revised Figure 2 and
Table 2 under `publication_assets/`. It contains no committed `outputs/`
directory and no bulky raw-result, checkpoint, or bootstrap files.

## Recommended: Git command line

1. Extract the ZIP into an empty folder.
2. Open Git Bash or a terminal in that folder.
3. Run:

```bash
git init -b main
git add .
git commit -m "Add EGMS-Drive reproducibility source and revised Study 1 assets"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

If the remote repository already contains an initial README commit, fetch it
before pushing or create a new empty repository without starter files.

## GitHub web uploader

GitHub's Code page does not expand ZIP archives. Extract this archive first,
then choose **Add file → Upload files** and upload the extracted files/folders.
The curated release is intentionally kept below 100 files, and every file
is below the web uploader's 25 MiB per-file limit.

## Verify before upload

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python tools/build_complete_source_document.py --check
python -m compileall -q src tools tests run_publication.py run_study1.py run_study1r2.py run_studies.py
python run_publication.py --output outputs/github_check
EGMS_PUBLICATION_OUTPUT="$PWD/outputs/github_check" \
  python -m unittest discover -s tests -v
```

Generated files stay under `outputs/`, which is ignored by Git. Do not commit
that directory unless you intentionally create a separate GitHub Release
asset.

## Release-manifest verification

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path('.')
manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
for relative, expected in manifest['files'].items():
    payload = (root / relative).read_bytes()
    assert len(payload) == expected['bytes'], relative
    assert hashlib.sha256(payload).hexdigest() == expected['sha256'], relative
print('PASS:', len(manifest['files']), 'files verified')
PY
```

The manifest intentionally does not include itself. Rebuild the archive after
changing source files so its hashes remain current.
