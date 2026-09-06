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
│   └── s10_figures_summary.py
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
2. Run the analysis scripts in order:
   ```bash
   python src/s02_features.py
   python src/s03_shot_quality.py
   python src/s01b_data_continuity.py
   python src/s04_profiles.py
   python src/s05_clustering.py
   python src/s06_stability.py
   python src/s07_load_analysis.py
   python src/s08_prediction.py
   python src/s09_robustness.py
   python src/s10_figures_summary.py
   ```

The scripts cover feature engineering, shot-quality modelling, the provider
taxonomy audit, team and player profile construction, clusterability tests,
season-to-season stability, competitive-load models, early-season prediction,
robustness checks, and final figure and summary generation.

Intermediate analytical frames are regenerated in `data/processed_release/`.
Tables, figures, model artefacts, logs, and `RESULTS_SUMMARY.md` are written to
`outputs/`.

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

`shots_clean.parquet` contains the cleaned shot sample after inclusion criteria,
event-field harmonisation, and score-state reconstruction. `matches.parquet`
contains the league matches entering the study. `fixtures.parquet` contains
domestic league, domestic cup, and European fixtures used to construct
calendar-load variables.

The original SofaScore-derived database and raw scraper outputs are not
included. The included parquet files are the analysis-ready inputs required to
run the reproducibility pipeline from step 2 onward.

---

## Citation

> *Will be updated upon publication.*

---

## License

Code released under the [MIT License](LICENSE).
