"""Step 11 - Supplementary appendix with complete statistical output.

Sports Medicine and Fitness guidelines ask for the original complete output
created by the statistical software to be supplied as an appendix. This script
collects the reproducible output files produced by the pipeline into one
submission-ready Markdown supplement:

    outputs/SUPPLEMENTARY_STATISTICAL_APPENDIX.md

The appendix deliberately preserves tables as raw CSV blocks and logs as raw
stdout blocks, rather than rewriting them as narrative results.
"""

from __future__ import annotations

from pathlib import Path

from common import LOGS, MODELS, OUT, TABLES, header

SCRIPT_LOG_NAMES = [
    ("s02_features", "s02"),
    ("s03_shot_quality", "s03"),
    ("s01b_data_continuity", "s01b"),
    ("s04_profiles", "s04"),
    ("s05_clustering", "s05"),
    ("s06_stability", "s06"),
    ("s07_load_analysis", "s07"),
    ("s08_prediction", "s08"),
    ("s09_robustness", "s09"),
    ("s10_figures_summary", "s10"),
    ("s12_core_validation", "s12"),
]

OPTIONAL_SOURCE_EXTRACTION = ("s01_extract_audit", "s01")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").rstrip()


def fenced(lines: list[str], language: str, body: str) -> None:
    lines.append(f"```{language}")
    lines.append(body if body else "[empty file]")
    lines.append("```")


def add_file_block(lines: list[str], level: int, path: Path, language: str) -> None:
    title = "#" * level
    rel = path.relative_to(OUT)
    lines.append(f"\n{title} `{rel}`\n")
    fenced(lines, language, read_text(path))


def main() -> None:
    header("STEP 11  Supplementary statistical appendix")

    lines: list[str] = [
        "# Supplementary Statistical Appendix",
        "",
        "Complete statistical output for *Longitudinal Shot-Quality and "
        "Attacking Performance Profiles in Czech Professional Football*.",
        "",
        "This appendix is generated directly from `outputs/` by "
        "`src/s11_statistical_appendix.py`. CSV result tables are reproduced "
        "verbatim in fenced `csv` blocks. Captured software logs are reproduced "
        "verbatim in fenced `text` blocks.",
        "",
        "## Output Inventory",
        "",
    ]

    tables = sorted(TABLES.glob("*.csv"))
    model_text = sorted(
        p for p in MODELS.glob("*") if p.suffix.lower() in {".csv", ".json", ".txt", ".tsv"}
    )
    logs = {p.stem: p for p in sorted(LOGS.glob("*.log"))}

    def log_for(script: str, legacy: str) -> Path | None:
        return logs.get(script) or logs.get(legacy)

    optional_source_log = log_for(*OPTIONAL_SOURCE_EXTRACTION)
    missing_logs = [
        script for script, legacy in SCRIPT_LOG_NAMES
        if log_for(script, legacy) is None
    ]
    missing_line = ", ".join(missing_logs) if missing_logs else "none"

    selected_logs: list[tuple[str, Path | None]] = []
    used_log_stems: set[str] = set()
    for script, legacy in SCRIPT_LOG_NAMES:
        path = log_for(script, legacy)
        selected_logs.append((script, path))
        if path is not None:
            used_log_stems.add(path.stem)

    known_log_stems = {
        name
        for pair in SCRIPT_LOG_NAMES + [OPTIONAL_SOURCE_EXTRACTION]
        for name in pair
    }
    extra_logs = [
        p for name, p in logs.items()
        if name not in used_log_stems
        and name not in known_log_stems
        and name not in {"s11", "s11_statistical_appendix"}
    ]

    lines.append(f"- Result tables included: {len(tables)}")
    lines.append(f"- Model-output text files included: {len(model_text)}")
    n_logs = len(used_log_stems) + len(extra_logs) + int(optional_source_log is not None)
    lines.append(f"- Captured stdout logs included: {n_logs}")
    if missing_logs:
        lines.append(f"- Pipeline logs not present at generation time: {missing_line}")
    if optional_source_log is None:
        lines.append(
            "- Optional source-database extraction: not included; reproduction "
            "starts from the released processed data."
        )
    else:
        lines.append("- Optional source-database extraction log included.")

    lines.append("\n## Complete CSV Result Tables\n")
    for path in tables:
        add_file_block(lines, 3, path, "csv")

    if model_text:
        lines.append("\n## Model Output Files\n")
        for path in model_text:
            language = "json" if path.suffix == ".json" else "csv" if path.suffix == ".csv" else "text"
            add_file_block(lines, 3, path, language)

    lines.append("\n## Captured Statistical Software Logs\n")
    lines.append(
        "`s01_extract_audit.py` is an optional source-database extraction step "
        "and is not required to reproduce the analyses from the released "
        "processed data."
    )
    if optional_source_log is not None:
        add_file_block(lines, 3, optional_source_log, "text")

    for script, path in selected_logs:
        if path is None:
            lines.append(f"\n### `{script}.log`\n")
            lines.append("[log file not present; rerun the pipeline with the README logging command]")
            continue
        add_file_block(lines, 3, path, "text")

    if extra_logs:
        lines.append("\n## Additional Logs\n")
        for path in extra_logs:
            add_file_block(lines, 3, path, "text")

    out = OUT / "SUPPLEMENTARY_STATISTICAL_APPENDIX.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"  [doc  ] {out.relative_to(OUT.parent)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
