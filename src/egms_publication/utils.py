from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].astype(str).eq(str(value))
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    for source, replacement in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(source, replacement)
    return text


def write_table(frame: pd.DataFrame, stem: Path) -> dict[str, Path]:
    """Write the same data to CSV, Markdown, and dependency-free LaTeX."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("csv", "md", "tex")}
    frame.to_csv(paths["csv"], index=False)
    def markdown_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", r"\|").replace("\n", " ")

    markdown_lines = [
        "| " + " | ".join(markdown_cell(column) for column in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(markdown_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    paths["md"].write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    alignment = "l" * len(frame.columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\hline"]
    lines.append(" & ".join(_latex_escape(column) for column in frame.columns) + r" \\")
    lines.append("\\hline")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    paths["tex"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
