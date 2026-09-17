# Czech Football Shot-Quality Profiles
### Longitudinal Shot Quality and Attacking Performance Profiles in the Czech First League

---

## Repository Structure

```
czech-football-shot-quality-profiles/
├── data/
│   └── processed_release/
│       ├── matches.parquet       # League matches entering the study
│       ├── shots_clean.parquet   # Analysis-ready shot sample from the extraction step
│       ├── shots_features.parquet # Engineered shot-level features
│       ├── team_match_load.parquet # Match-level calendar-load features
│       └── fixtures.parquet      # Team fixtures used for calendar-load features
├── src/
│   ├── s01_extract_audit.py      # Optional rebuild from the local MySQL database
│   ├── s02_features.py
│   ├── s03_shot_quality.py
│   ├── s01b_data_continuity.py
│   ├── s04_profiles.py
│   ├── s05_clustering.py
│   ├── s06_stability.py
│   ├── s07_load_analysis.py
│   ├── s08_prediction.py
│   ├── s09_robustness.py
│   ├── s10_figures_summary.py
│   ├── s11_statistical_appendix.py
│   └── s12_core_validation.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Reproducing the Results

The statistical analyses can be reproduced from the processed input files
included in this repository without access to the original scraper database.

1. Install dependencies:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the analysis scripts in order, capturing the complete stdout logs:
   ```bash
   mkdir -p outputs/logs
   set -o pipefail
   for s in src/s02_features.py src/s03_shot_quality.py \
            src/s01b_data_continuity.py src/s04_profiles.py src/s05_clustering.py \
            src/s06_stability.py src/s07_load_analysis.py src/s08_prediction.py \
            src/s09_robustness.py src/s12_core_validation.py \
            src/s10_figures_summary.py src/s11_statistical_appendix.py; do
       name="$(basename "$s" .py)"
       python "$s" 2>&1 | tee "outputs/logs/${name}.log"
       rc="${PIPESTATUS[0]}"
       [ "$rc" -eq 0 ] || exit "$rc"
   done
   ```

The scripts cover feature engineering, shot-quality modelling, the provider
taxonomy audit, team and player profile construction, clusterability tests,
season-to-season stability, competitive-load models, non-overlapping
early-season prediction, robustness checks, core validation analyses, final
figure and summary generation, and the supplementary statistical appendix.

Intermediate analytical frames are regenerated in `data/processed_release/`.
Tables, figures, model artefacts, captured logs, `RESULTS_SUMMARY.md`, and
`SUPPLEMENTARY_STATISTICAL_APPENDIX.md` are written to `outputs/`.

---

## Optional Database Rebuild

`src/s01_extract_audit.py` rebuilds the processed input files from a local
MySQL database named `czech_soccer`, populated by the companion scraper used in
the project workspace. The public release does not include that database.

Default connection settings are `localhost:8889`, user `root`, password
`root`. They can be overridden with `DB_HOST`, `DB_PORT`, `DB_USER`,
`DB_PASSWORD`, and `DB_NAME`.

If the database is available, run this first:

```bash
python src/s01_extract_audit.py
```

The two extraction-audit tables produced by that step are included in
`outputs/tables/` so the downstream summary can be regenerated from the
processed release files.

---

## Notes on the Data

The analytical sample covers Czech First League seasons 2022/23, 2023/24, and
2024/25. Season 2025/26 is excluded by design because it was incomplete in the
source database at analysis time and would have biased early-season analyses.

`processed_release/` contains five parquet files. `shots_clean.parquet` is the
cleaned shot sample after inclusion criteria, event-field harmonisation, and
score-state reconstruction; `shots_features.parquet` adds engineered spatial,
technical, context, and calendar-load features; `team_match_load.parquet`
contains team-match calendar-load variables; `matches.parquet` contains the
league matches entering the study; and `fixtures.parquet` contains domestic
league, domestic cup, and European fixtures used to construct calendar-load
variables.

Early, remainder, and split-half profile aggregates require at least 20 shots
in the relevant window. This avoids treating very small early windows as stable
team profiles; for example, the three-match stabilisation table has `n = 47`
because FC Hradec Kralove 2024/25 had only 15 shots in its first three matches.

The original SofaScore-derived database and raw scraper outputs are not
included. The included parquet files are the analysis-ready inputs required to
run the reproducibility pipeline from step 2 onward.

File names such as `figure1_shot_density.pdf` and `table7_robustness.csv` are
internal pipeline slugs. Use the manuscript captions and the journal's final
numbering when preparing the submission package.

---

## Citation

> *Will be updated upon publication.*

---

## License

Code released under the [MIT License](LICENSE).
