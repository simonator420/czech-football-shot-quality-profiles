# Results summary

Longitudinal shot-quality and attacking-performance profiles in the Czech First League, seasons 2022/23-2024/25.

Season 2025/26 is excluded from the study by design.


## Sample

- 21,311 shots from 826 matches, 647 players, 16 teams per season.
- Overall conversion 10.67%, 25.8 shots per match.
- Chronological validation: train 2022/23 + 2023/24, held-out test 2024/25.
- Early-window profile estimates require at least 20 shots in the window; this avoids treating very small early samples as stable profile estimates.

## Shot-quality model (RQ6, proposal 5.3)

- Main model: **CatBoost (depth=4,l2=10)**.
- Held-out AUC **0.7747**, Brier score 0.0777, Brier skill score **0.1461** against the overall goal-rate baseline (log loss 0.3279).
- Calibration slope 0.9805, intercept -0.3199, ECE 0.0269.
- Temporal validation: Temporal development validation: 2022/23 -> 2023/24: AUC 0.788, calibration intercept -0.1884; Final temporal holdout: 2022/23 + 2023/24 -> 2024/25: AUC 0.7747, calibration intercept -0.3199.
- Out-of-fold shot quality (5-fold, grouped by match) supplies the shot-quality value used in every downstream analysis, so no team is profiled with predictions from its own matches.

## Data-quality finding (step 1b)

- The provider's `situation` taxonomy changes abruptly at the 2024/25 season boundary: fast-break share rises 3.1x and throw-in set-piece share 5.2x, with **non-overlapping monthly ranges** before and after (Welch p = 5e-09 and 5e-08).
- Consequence: profile features are standardised **within season**, and the shot-quality and profile-construction workflow is re-run without the `situation` predictor or situation-derived profile features (Table 7 and Table 7b).

## Hypothesis 1 - discrete attacking-profile types

- **NOT SUPPORTED**. Four independent checks agree that the profile space is a continuum rather than a set of discrete types:
  1. Gap statistic selects **k = 1** for both feature blocks.
  2. Across all 36 candidate solutions the best minimum bootstrap Jaccard is **0.621**. 0 solutions reach the 0.75 threshold for a stable cluster and 2 reach the 0.60 threshold for a reportable pattern, so no partition is stable in Hennig's sense.
  3. A null reference drawn with the same covariance but no clusters by construction reaches silhouette 0.300 against the observed 0.418, and mean bootstrap Jaccard 0.560 against the observed 0.600 - the observed solution is barely separable from data known to contain no clusters.
  4. Removing feature redundancy does not restore stability, so the instability is not an artefact of correlated inputs.
- A 2-cluster partition is retained as a descriptive device only, with its stability reported alongside it.

## RQ6 - separating chance creation from finishing

- **SUPPORTED**. The profile space resolves into two orthogonal axes (r = 0.000): a chance-creation axis loading on shot quality, distance and high-quality share, and a finishing axis loading on goal rate, on-target rate and goals minus expected goals.

## Hypothesis 2 - season-to-season stability

- Chance creation axis: year-to-year r = **0.764** [0.56, 0.88], ICC 0.78.
- Finishing axis: year-to-year r = **0.232** [-0.14, 0.55], ICC 0.205.
- Chance creation persists across seasons; finishing does not, which is the pattern hypothesis 5 predicts and the reason finishing should not be read as a stable team trait.
- Split-half reliability is descriptively consistent with the process/finishing distinction: xG per match first-half vs second-half r = **0.751** [0.572, 0.868], while goals-minus-xG r = **0.452** [0.124, 0.665].
- Paired team-cluster bootstrap difference in split-half r (chance_creation_axis minus finishing_axis): 0.076 [-0.354, 0.282].
- Paired team-cluster bootstrap difference in split-half r (xg_per_match minus goals_minus_xg_per_match): 0.299 [-0.039, 0.618].

## Hypothesis 3 - competitive load and shot quality

- Short rest (<= 3 days vs >= 4): effect on Shot quality (logit) = **-0.0342** [-0.0909, 0.0225], p = 0.2369.
- Matches in previous 14 days (per match): effect on Shot quality (logit) = **0.0005** [-0.0177, 0.0187], p = 0.9557.
- Days since previous fixture (per day, capped at 14): effect on Shot quality (logit) = **0.0061** [0.0002, 0.0121], p = 0.0442.
- Secondary outcomes surviving BH-FDR correction: 1 of 21.
- Short rest is rare in this league (about 6% of shots), so the design is better powered to detect congestion effects through the continuous matches-in-14-days exposure than through the short-rest contrast.

## RQ3 - early-season prediction

- Primary model specification: ridge regression. For the pre-specified first-10-match window predicting remainder-season xG per match, R2 = **0.297**, MAE 0.291, Spearman rho 0.618.
- Naive early-value baseline for the same target/window: R2 = **0.377**, MAE 0.303, Spearman rho 0.726. Random forests are reported only as sensitivity comparisons.
- For the first-10-match chance-creation axis, ridge R2 = **-0.627** against the non-overlapping season remainder.
- All early-season prediction targets exclude the matches used as predictors, so the estimates are not inflated by part-whole overlap.
- For the 2024/25 holdout, shot-quality aggregates use the train-only `shot_quality_holdout` predictions and profile-axis PCA/scaling is fit only on 2022/23-2023/24.

## Practical validity

- Exploratory concurrent associations: process-only chance creation is associated with points per match (beta 0.1732, team-cluster bootstrap CI [0.132, 0.201], p < .001) and goal difference per match (beta 0.2745, CI [0.212, 0.317], p < .001) after adjusting for finishing and season.

## Files

| Output | Content |
| --- | --- |
| `tables/table1_dataset_construction.csv` | Inclusion flow from scraped matches to the analytical sample |
| `tables/table2_sample_characteristics.csv` | Sample characteristics by season |
| `tables/table2b_taxonomy_continuity.csv` | Cross-season continuity of the event taxonomy |
| `tables/table3_shot_quality_models.csv` | Shot-quality model comparison and calibration |
| `tables/table3b_shot_quality_by_subset.csv` | Model performance by shot subset |
| `tables/table3c_temporal_validation.csv` | Temporal development and final holdout shot-quality validation |
| `tables/table4_cluster_characteristics.csv` | Cluster centroids, stability and representative teams |
| `tables/table4b_cluster_selection_metrics.csv` | Validity indices for all 36 candidate solutions |
| `tables/table4f_gap_statistic.csv` | Gap statistic, including k = 1 |
| `tables/table5_transition_matrix.csv` | Season-to-season cluster transitions |
| `tables/table5b_profile_repeatability.csv` | Year-to-year repeatability by feature |
| `tables/table5c_trajectory_typology.csv` | Multi-season trajectory classification |
| `tables/table6_load_effects.csv` | Competitive-load effects on shot quality and selection |
| `tables/table7_robustness.csv` | All sensitivity and robustness analyses |
| `tables/table7b_no_situation_profile_comparison.csv` | Team-season profile comparison after rebuilding the no-situation workflow |
| `tables/table8_split_half_reliability.csv` | Split-half reliability of process and finishing indicators |
| `tables/table8b_split_half_differences.csv` | Bootstrap contrasts between chance-creation and finishing reliability |
| `tables/table9_practical_validity.csv` | Process-only chance creation and team results |
| `tables/table9b_opponent_adjusted_profiles.csv` | Opponent-adjusted attacking ratings |
| `tables/tableS2_stabilisation_curves.csv` | Non-overlapping early-versus-remainder stabilisation curves |
| `tables/tableS3_early_season_regression.csv` | Early-season prediction of remainder-season profiles |
| `figures/figure1_shot_density.pdf` | Shot density and modelled shot quality by location |
| `figures/figure2_calibration.pdf` | Calibration on the held-out season |
| `figures/figure3_shap_importance.pdf` | SHAP feature contributions |
| `figures/figure4_profile_space.pdf` | Continuous profile space and UMAP projection |
| `figures/figure6_profile_transitions.pdf` | Transitions, persistence and repeatability |
| `figures/figure7_shot_quality_by_rest.pdf` | Shot quality by rest interval and congestion |
| `figures/figure8_late_game_trends.pdf` | Shot quality across the match |
| `figures/figure9_team_resilience.pdf` | Team resilience under congestion |
| `figures/figure10_split_half_reliability.pdf` | Split-half reliability of chance creation and finishing |
| `figures/figure11_practical_validity.pdf` | Process-only chance creation and team results |
| `figures/figureS1_stabilisation.pdf` | Early-versus-remainder stabilisation curves |
