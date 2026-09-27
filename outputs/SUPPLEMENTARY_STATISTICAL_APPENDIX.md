# Supplementary Statistical Appendix

Complete statistical output for *Longitudinal Shot-Quality and Attacking Performance Profiles in Czech Professional Football*.

This appendix is generated directly from `outputs/` by `src/s11_statistical_appendix.py`. CSV result tables are reproduced verbatim in fenced `csv` blocks. Captured software logs are reproduced verbatim in fenced `text` blocks.

## Output Inventory

- Result tables included: 30
- Model-output text files included: 2
- Captured stdout logs included: 11
- Optional source-database extraction: not included; reproduction starts from the released processed data.

## Complete CSV Result Tables


### `tables/table1_dataset_construction.csv`

```csv
Step,Matches,Shots,Note
"Matches scraped, seasons 2022/23-2024/25",866,21644,Season 2025/26 excluded from the study by design
Finished matches,841,21644,Postponed/cancelled fixtures removed
Matches with shot-map data,836,21644,5 finished matches had no shot map available
Czech First League matches (regular season + championship/relegation group),826,21373,10 relegation/promotion play-off matches vs second-tier opposition excluded from the shot sample (retained as calendar load)
Duplicate shot records removed,826,21373,0 duplicates
Penalty shoot-out attempts removed,826,21373,0 shoot-out attempts
Own goals removed,826,21311,62 own goals (not attempts on the opponent's goal)
Shots with valid spatial coordinates,826,21311,0 shots without coordinates
Coordinates within the normalised pitch grid,826,21311,0 shots with out-of-range coordinates
Shots with valid team and player identifiers,826,21311,0 shots without team/player link
Final analytical shot sample,826,21311,0 shots with missing shot type/body part/situation
```

### `tables/table2_sample_characteristics.csv`

```csv
Season,Matches,Teams,Shots,Shots per match,Goals,Goals per match,Conversion rate (%),On-target (%),Blocked (%),Headers (%),Penalties,Set-piece origin (%),Distinct players
2022/23,275,16,6948,25.27,792,2.88,11.4,35.1,24.6,19.7,74,26.8,377
2023/24,276,16,7261,26.31,764,2.77,10.52,33.7,26.0,18.4,80,27.6,398
2024/25,275,16,7102,25.83,718,2.61,10.11,32.8,27.1,18.7,69,32.7,426
All seasons,826,18,21311,25.8,2274,2.75,10.67,33.9,25.9,18.9,223,29.1,647
```

### `tables/table2b_taxonomy_continuity.csv`

```csv
Variable,Type,2022/23,2023/24,2024/25,Ratio 2024/25 : earlier seasons
situation = assisted,Technical annotation,0.5574,0.5452,0.4678,0.85
situation = corner,Technical annotation,0.155,0.1592,0.165,1.05
situation = open_play,Technical annotation,0.1586,0.149,0.1349,0.88
situation = set_piece,Technical annotation,0.0635,0.0694,0.0684,1.03
situation = throw_in,Technical annotation,0.0112,0.0139,0.0651,5.18
situation = free_kick,Technical annotation,0.0278,0.0227,0.0189,0.75
situation = fast_break,Technical annotation,0.0158,0.0295,0.0703,3.1
situation = penalty,Technical annotation,0.0107,0.011,0.0097,0.9
mean_distance_m,Spatial / derived,18.1188,18.0029,17.0413,0.94
close_range_share,Spatial / derived,0.1982,0.2125,0.262,1.28
inside_box_share,Spatial / derived,0.6186,0.6063,0.6322,1.03
header_share,Spatial / derived,0.1966,0.184,0.1867,0.98
```

### `tables/table3_shot_quality_models.csv`

```csv
Model,N shots,AUC,PR-AUC,Log loss,Brier score,Brier skill score,Calibration slope,Calibration intercept,ECE,Main model
Model 0: overall goal-rate baseline,7102,,,0.3279,0.0909,0.0,,,0.0084,
Model 1: spatial logistic regression,7102,0.7327,0.2774,0.2985,0.0845,0.0712,0.9655,-0.2338,0.0233,
Model 2: contextual logistic regression,7102,0.7717,0.3583,0.2804,0.0787,0.1351,0.9582,-0.2962,0.0277,
Regularised logistic regression (C=1.0),7102,0.772,0.379,0.2788,0.0782,0.14,0.9388,-0.3229,0.0281,
"Random forest (depth=8,leaf=20)",7102,0.7701,0.3675,0.2777,0.0776,0.1465,1.0624,-0.2791,0.0241,
"XGBoost (depth=3,mcw=5)",7102,0.7753,0.3815,0.2769,0.0778,0.145,1.0143,-0.3281,0.0282,
"CatBoost (depth=4,l2=10)",7102,0.7747,0.383,0.2764,0.0777,0.1461,0.9805,-0.3199,0.0269,yes
```

### `tables/table3b_shot_quality_by_subset.csv`

```csv
Model,N shots,AUC,PR-AUC,Log loss,Brier score,Brier skill score,Calibration slope,Calibration intercept,ECE
"Test season, penalties excluded",7033,0.7596,0.3337,0.2734,0.0765,0.111,0.978,-0.3192,0.0266
"Test season, all shots",7102,0.7747,0.383,0.2764,0.0777,0.1461,0.9805,-0.3199,0.0269
"Test season, open play only",4779,0.7562,0.3451,0.2854,0.0808,0.1165,0.95,-0.2459,0.021
```

### `tables/table3c_temporal_validation.csv`

```csv
Model,N shots,AUC,PR-AUC,Log loss,Brier score,Brier skill score,Calibration slope,Calibration intercept,ECE
Temporal development validation: 2022/23 -> 2023/24,7261,0.788,0.4133,0.2746,0.0773,0.1797,1.0602,-0.1884,0.0158
Final temporal holdout: 2022/23 + 2023/24 -> 2024/25,7102,0.7747,0.383,0.2764,0.0777,0.1461,0.9805,-0.3199,0.0269
```

### `tables/table4_cluster_characteristics.csv`

```csv
Cluster,Profile,Team-seasons,Bootstrap Jaccard,Stability verdict,mean_shot_quality,mean_shot_quality (z),median_shot_quality,median_shot_quality (z),xg_per_match,xg_per_match (z),goals_minus_xg_per_match,goals_minus_xg_per_match (z),mean_distance_m,mean_distance_m (z),mean_angle_proxy,mean_angle_proxy (z),high_quality_share,high_quality_share (z),low_quality_share,low_quality_share (z),shots_per_match,shots_per_match (z),on_target_rate,on_target_rate (z),blocked_rate,blocked_rate (z),goal_rate,goal_rate (z),header_share,header_share (z),assisted_share,assisted_share (z),set_piece_share,set_piece_share (z),close_range_share,close_range_share (z),fast_break_share,fast_break_share (z),Representative team-seasons
0,High-volume chance creators,8,0.6,pattern,0.128,1.82,0.079,1.78,2.082,1.92,0.075,0.49,16.573,-1.61,0.595,0.12,0.249,1.8,0.315,-1.66,16.239,1.63,0.361,0.73,0.244,-0.53,0.134,1.37,0.211,0.69,0.535,0.16,0.287,-0.01,0.263,1.32,0.037,0.25,"AC Sparta Praha 2022/23, AC Sparta Praha 2023/24, FC Viktoria Plzeň 2022/23"
1,Main body of the league,40,0.604,pattern,0.105,-0.36,0.063,-0.36,1.281,-0.38,-0.064,-0.1,18.035,0.32,0.588,-0.02,0.176,-0.36,0.411,0.33,12.22,-0.33,0.332,-0.15,0.263,0.11,0.099,-0.27,0.183,-0.14,0.52,-0.03,0.292,0.0,0.214,-0.26,0.039,-0.05,"MFK Karviná 2024/25, FC Zlín 2022/23, Bohemians Praha 1905 2022/23"
```

### `tables/table4b_cluster_selection_metrics.csv`

```csv
Feature set,Algorithm,k,Silhouette,Calinski-Harabasz,Davies-Bouldin,BIC (GMM),Mean bootstrap Jaccard,Min bootstrap Jaccard,Smallest cluster n
Full (17 features),k-means,2,0.4176,27.084,0.8878,,0.6021,0.6003,8
Full (17 features),k-means,3,0.216,21.3807,1.5133,,0.51,0.4497,8
Full (17 features),k-means,4,0.197,18.7975,1.3887,,0.4457,0.3573,8
Full (17 features),k-means,5,0.2048,16.7494,1.3612,,0.4152,0.3431,8
Full (17 features),k-means,6,0.2008,15.6609,1.2073,,0.4184,0.3158,2
Full (17 features),k-means,7,0.2054,15.0514,1.2221,,0.4289,0.2813,2
Full (17 features),GMM,2,0.4176,27.084,0.8878,916.7834,0.5359,0.5006,8
Full (17 features),GMM,3,0.1339,16.1823,1.8035,939.6123,0.3981,0.2018,7
Full (17 features),GMM,4,0.0752,12.5895,2.1952,953.4022,0.352,0.2224,6
Full (17 features),GMM,5,0.1244,12.0622,1.4861,971.8134,0.3564,0.2598,2
Full (17 features),GMM,6,0.0912,10.3755,1.2738,963.9925,0.3439,0.216,1
Full (17 features),GMM,7,0.1329,10.9806,1.3648,991.0981,0.3214,0.2512,1
Full (17 features),Ward,2,0.4176,27.084,0.8878,,0.6263,0.6205,8
Full (17 features),Ward,3,0.1748,19.0488,1.7604,,0.467,0.3779,8
Full (17 features),Ward,4,0.1788,16.4879,1.551,,0.4159,0.3268,7
Full (17 features),Ward,5,0.195,15.4613,1.3767,,0.397,0.3048,7
Full (17 features),Ward,6,0.1994,14.742,1.2466,,0.4299,0.3165,2
Full (17 features),Ward,7,0.2185,14.555,1.1539,,0.4597,0.3132,2
Reduced (9 features),k-means,2,0.1981,14.7272,1.678,,0.5032,0.493,22
Reduced (9 features),k-means,3,0.193,12.8867,1.5953,,0.4283,0.417,7
Reduced (9 features),k-means,4,0.1909,12.1656,1.3842,,0.4042,0.3611,7
Reduced (9 features),k-means,5,0.1973,11.7327,1.3694,,0.3956,0.2996,7
Reduced (9 features),k-means,6,0.232,11.8552,1.1721,,0.4474,0.3927,4
Reduced (9 features),k-means,7,0.2536,12.2543,1.1128,,0.4771,0.4273,2
Reduced (9 features),GMM,2,0.1825,9.3393,1.8707,821.1149,0.3788,0.2712,12
Reduced (9 features),GMM,3,0.0987,8.064,2.0969,840.4752,0.3127,0.2546,9
Reduced (9 features),GMM,4,0.0355,5.2428,2.2356,853.1756,0.2847,0.2222,5
Reduced (9 features),GMM,5,0.1368,7.5168,1.2171,868.474,0.3005,0.1899,1
Reduced (9 features),GMM,6,0.1161,6.978,1.4226,881.4512,0.3225,0.268,1
Reduced (9 features),GMM,7,0.2161,10.2786,1.1121,885.2924,0.389,0.323,1
Reduced (9 features),Ward,2,0.1927,13.7894,1.7145,,0.4379,0.4093,19
Reduced (9 features),Ward,3,0.1633,11.3397,1.5941,,0.4004,0.388,7
Reduced (9 features),Ward,4,0.1672,11.2636,1.5368,,0.4,0.3346,7
Reduced (9 features),Ward,5,0.1982,11.2425,1.4047,,0.4219,0.345,4
Reduced (9 features),Ward,6,0.1998,11.1124,1.2724,,0.4255,0.3664,4
Reduced (9 features),Ward,7,0.226,11.4644,1.1556,,0.4712,0.3842,2
```

### `tables/table4c_cluster_membership.csv`

```csv
team_id,team_name,season_name,team_season,cluster,profile,cluster_stability,chance_creation_axis,finishing_axis,chance_source_axis,matches,shots,goals,mean_shot_quality,median_shot_quality,xg_per_match,goals_minus_xg_per_match,mean_distance_m,mean_angle_proxy,high_quality_share,low_quality_share,shots_per_match,on_target_rate,blocked_rate,goal_rate,header_share,assisted_share,set_piece_share,close_range_share,fast_break_share
1,FC Viktoria Plzeň,2022/23,FC Viktoria Plzeň 2022/23,0,High-volume chance creators,0.6003,3.5251,-3.0069,-2.33,35,549,60,0.1083,0.0703,1.6991,0.0152,17.02,0.5792,0.1949,0.3461,15.6857,0.3115,0.2532,0.1093,0.2495,0.5774,0.2623,0.2659,0.0091
1,FC Viktoria Plzeň,2023/24,FC Viktoria Plzeň 2023/24,0,High-volume chance creators,0.6003,5.742,-0.4599,-0.0122,35,538,75,0.1324,0.0841,2.0345,0.1084,16.1411,0.5937,0.2658,0.2918,15.3714,0.3587,0.2658,0.1394,0.2175,0.5595,0.2807,0.2825,0.0316
1,FC Viktoria Plzeň,2024/25,FC Viktoria Plzeň 2024/25,0,High-volume chance creators,0.6003,4.7025,-1.1342,-0.3807,35,511,70,0.1398,0.0941,2.0418,-0.0418,15.6134,0.54,0.2857,0.2524,14.6,0.3346,0.2857,0.137,0.2211,0.4697,0.3268,0.2975,0.0705
2,SK Sigma Olomouc,2022/23,SK Sigma Olomouc 2022/23,1,Main body of the league,0.604,-0.1091,-0.4709,0.8245,35,447,50,0.0967,0.0607,1.2356,0.193,17.677,0.611,0.123,0.4004,12.7714,0.3512,0.2237,0.1119,0.2192,0.5526,0.3087,0.1678,0.0112
2,SK Sigma Olomouc,2023/24,SK Sigma Olomouc 2023/24,1,Main body of the league,0.604,0.4838,1.2253,1.3905,32,334,40,0.1029,0.0664,1.0739,0.1761,17.6435,0.5707,0.1677,0.3892,10.4375,0.3982,0.2186,0.1198,0.1707,0.5569,0.2994,0.2186,0.015
2,SK Sigma Olomouc,2024/25,SK Sigma Olomouc 2024/25,1,Main body of the league,0.604,0.1381,0.9331,2.82,35,371,48,0.1265,0.0693,1.3406,0.0308,17.3425,0.5775,0.2237,0.3504,10.6,0.3504,0.2561,0.1294,0.1644,0.4016,0.3612,0.2453,0.0809
3,AC Sparta Praha,2022/23,AC Sparta Praha 2022/23,0,High-volume chance creators,0.6003,4.9108,-1.934,0.785,35,573,74,0.1209,0.0659,1.9798,0.1345,16.8526,0.6022,0.2269,0.3665,16.3714,0.3508,0.2286,0.1291,0.2373,0.4956,0.3368,0.2164,0.0227
3,AC Sparta Praha,2023/24,AC Sparta Praha 2023/24,0,High-volume chance creators,0.6003,5.6119,1.9829,1.5424,35,559,82,0.1285,0.0789,2.0522,0.2907,16.6531,0.6314,0.2397,0.3113,15.9714,0.4043,0.2308,0.1467,0.1968,0.542,0.2701,0.2361,0.0376
3,AC Sparta Praha,2024/25,AC Sparta Praha 2024/25,1,Main body of the league,0.604,0.8383,1.2138,-0.0922,35,569,61,0.1106,0.0791,1.7984,-0.0555,16.7788,0.606,0.2197,0.3251,16.2571,0.3409,0.2917,0.1072,0.188,0.4868,0.3199,0.2232,0.0527
4,SK Slavia Praha,2022/23,SK Slavia Praha 2022/23,0,High-volume chance creators,0.6003,5.8015,3.0386,1.3589,35,537,94,0.1188,0.0706,1.8234,0.8623,17.5038,0.6429,0.2142,0.3333,15.3429,0.4041,0.2533,0.175,0.162,0.5866,0.2551,0.2142,0.0168
4,SK Slavia Praha,2023/24,SK Slavia Praha 2023/24,0,High-volume chance creators,0.6003,4.042,-0.8092,-1.7011,35,646,74,0.1253,0.0705,2.3126,-0.1983,16.987,0.5839,0.2601,0.3762,18.4571,0.3467,0.2415,0.1146,0.209,0.5666,0.2786,0.2539,0.0248
4,SK Slavia Praha,2024/25,SK Slavia Praha 2024/25,0,High-volume chance creators,0.6003,6.6182,-0.4075,-1.8894,35,634,75,0.1499,0.096,2.7152,-0.5723,15.8119,0.5829,0.306,0.2461,18.1143,0.3754,0.1924,0.1183,0.1956,0.4811,0.2871,0.3344,0.0836
5,FK Teplice,2022/23,FK Teplice 2022/23,1,Main body of the league,0.604,-0.9784,1.4157,0.6826,35,357,45,0.1021,0.0558,1.0419,0.2438,18.6496,0.5985,0.1513,0.4594,10.2,0.3641,0.2017,0.1261,0.1821,0.5602,0.2437,0.1653,0.0196
5,FK Teplice,2023/24,FK Teplice 2023/24,1,Main body of the league,0.604,-3.503,-1.0456,0.353,34,435,35,0.0889,0.0545,1.1379,-0.1084,19.5098,0.5754,0.1103,0.4667,12.7941,0.2989,0.2483,0.0805,0.1839,0.5011,0.3103,0.1747,0.0115
5,FK Teplice,2024/25,FK Teplice 2024/25,1,Main body of the league,0.604,-1.8325,-1.4499,0.2915,35,445,41,0.1026,0.0703,1.3047,-0.1333,17.6028,0.5304,0.1865,0.3708,12.7143,0.2921,0.2584,0.0921,0.2022,0.4292,0.3596,0.2607,0.0584
6,FC Zlín,2022/23,FC Zlín 2022/23,1,Main body of the league,0.604,-0.6763,-1.9952,-0.6718,35,424,42,0.0928,0.0581,1.1238,0.0762,17.7533,0.5698,0.1297,0.4458,12.1143,0.3561,0.2335,0.0991,0.2642,0.5613,0.2689,0.2288,0.0118
6,FC Zlín,2023/24,FC Zlín 2023/24,1,Main body of the league,0.604,-0.5014,-1.4661,-0.1642,35,395,40,0.098,0.062,1.1059,0.037,17.5358,0.5587,0.1696,0.438,11.2857,0.3114,0.2329,0.1013,0.2304,0.5671,0.2962,0.2506,0.0203
7,FC Zbrojovka Brno,2022/23,FC Zbrojovka Brno 2022/23,1,Main body of the league,0.604,-1.2645,-0.6925,-1.0549,35,402,39,0.0979,0.0623,1.1239,-0.0096,17.9225,0.6087,0.1393,0.3856,11.4857,0.306,0.2886,0.097,0.1592,0.5746,0.2512,0.1816,0.0075
8,FK Pardubice,2022/23,FK Pardubice 2022/23,1,Main body of the league,0.604,-2.5112,0.5468,-3.2785,34,395,32,0.0898,0.0591,1.043,-0.1018,19.0358,0.5978,0.1468,0.438,11.6176,0.3165,0.243,0.081,0.1671,0.643,0.1696,0.1848,0.0025
8,FK Pardubice,2023/24,FK Pardubice 2023/24,1,Main body of the league,0.604,-0.5074,-0.4001,0.9235,34,364,35,0.1048,0.0702,1.1225,-0.0931,17.8458,0.6059,0.1786,0.3544,10.7059,0.3269,0.2885,0.0962,0.1566,0.5165,0.294,0.2033,0.033
8,FK Pardubice,2024/25,FK Pardubice 2024/25,1,Main body of the league,0.604,-2.7532,0.5742,-0.6392,35,335,23,0.1095,0.0703,1.0485,-0.3913,17.3552,0.5802,0.203,0.3701,9.5714,0.3045,0.3104,0.0687,0.1493,0.4836,0.2985,0.2179,0.0776
9,FK Jablonec,2022/23,FK Jablonec 2022/23,1,Main body of the league,0.604,-1.2491,3.5132,0.387,35,419,48,0.0958,0.0573,1.1468,0.2247,18.9595,0.6348,0.1384,0.4535,11.9714,0.4057,0.2339,0.1146,0.1122,0.6086,0.2363,0.1504,0.0167
9,FK Jablonec,2023/24,FK Jablonec 2023/24,1,Main body of the league,0.604,-0.0438,0.5552,0.76,34,376,41,0.1049,0.0619,1.16,0.0459,18.0315,0.6021,0.1782,0.4069,11.0588,0.3484,0.2473,0.109,0.1862,0.5612,0.2846,0.2234,0.0319
9,FK Jablonec,2024/25,FK Jablonec 2024/25,1,Main body of the league,0.604,1.9333,0.8358,0.2714,35,498,59,0.1231,0.078,1.7519,-0.0662,16.9451,0.5812,0.2671,0.3293,14.2286,0.3635,0.261,0.1185,0.1948,0.4819,0.3353,0.2671,0.0602
10,Baník Ostrava,2022/23,Baník Ostrava 2022/23,1,Main body of the league,0.604,-0.0976,0.4085,-1.6731,34,462,51,0.1062,0.0538,1.4437,0.0563,18.7852,0.5962,0.1753,0.4524,13.5882,0.3333,0.2424,0.1104,0.1905,0.5996,0.2165,0.21,0.0195
10,Baník Ostrava,2023/24,Baník Ostrava 2023/24,1,Main body of the league,0.604,-2.0332,3.2789,-3.4146,35,558,53,0.0968,0.0538,1.5436,-0.0293,19.9291,0.5466,0.1577,0.4677,15.9429,0.3405,0.2832,0.095,0.1362,0.647,0.1953,0.138,0.0358
10,Baník Ostrava,2024/25,Baník Ostrava 2024/25,1,Main body of the league,0.604,1.5344,0.6447,-1.6338,34,537,55,0.1229,0.0805,1.9409,-0.3232,17.0471,0.5904,0.2272,0.3184,15.7941,0.324,0.2607,0.1024,0.1862,0.5251,0.3017,0.2737,0.0726
11,Bohemians Praha 1905,2022/23,Bohemians Praha 1905 2022/23,1,Main body of the league,0.604,0.4098,-0.053,3.6807,35,426,55,0.1019,0.059,1.2403,0.3311,18.1267,0.6391,0.1432,0.4272,12.1714,0.3779,0.2324,0.1291,0.2019,0.4272,0.331,0.1854,0.0235
11,Bohemians Praha 1905,2023/24,Bohemians Praha 1905 2023/24,1,Main body of the league,0.604,-1.3972,-3.1095,0.5417,35,427,34,0.1046,0.0618,1.2763,-0.3049,17.6621,0.556,0.1546,0.4028,12.2,0.3115,0.3021,0.0796,0.1944,0.466,0.3162,0.2225,0.0351
11,Bohemians Praha 1905,2024/25,Bohemians Praha 1905 2024/25,1,Main body of the league,0.604,-0.2644,-2.5609,-0.0627,34,451,36,0.1197,0.074,1.588,-0.5291,16.9133,0.5619,0.2173,0.3326,13.2647,0.2838,0.3149,0.0798,0.2151,0.4169,0.337,0.2772,0.1064
12,1. FC Slovácko,2022/23,1. FC Slovácko 2022/23,1,Main body of the league,0.604,-0.702,-1.7611,0.2506,35,391,39,0.0983,0.0556,1.0977,0.0166,18.0022,0.6149,0.1662,0.4425,11.1714,0.3248,0.2276,0.0997,0.2302,0.5294,0.289,0.2302,0.0077
12,1. FC Slovácko,2023/24,1. FC Slovácko 2023/24,1,Main body of the league,0.604,-0.9646,-0.403,-1.9916,35,445,43,0.1042,0.0638,1.325,-0.0964,18.4518,0.5345,0.1685,0.3978,12.7143,0.3101,0.2742,0.0966,0.1708,0.5483,0.2315,0.2157,0.0202
12,1. FC Slovácko,2024/25,1. FC Slovácko 2024/25,1,Main body of the league,0.604,-1.4747,-1.1962,-0.719,35,382,30,0.1193,0.0701,1.3025,-0.4453,17.5951,0.5676,0.1963,0.3717,10.9143,0.3063,0.288,0.0785,0.2016,0.4555,0.3089,0.2906,0.0707
13,FC Hradec Králové,2022/23,FC Hradec Králové 2022/23,1,Main body of the league,0.604,-0.8563,0.0756,-0.5,34,400,39,0.0975,0.0575,1.1467,0.0004,18.0622,0.6225,0.1575,0.4475,11.7647,0.35,0.255,0.0975,0.1775,0.5825,0.2475,0.2075,0.015
13,FC Hradec Králové,2023/24,FC Hradec Králové 2023/24,1,Main body of the league,0.604,0.4801,-2.481,0.8343,35,427,41,0.1094,0.0608,1.3344,-0.163,17.129,0.5835,0.2014,0.3958,12.2,0.3162,0.2529,0.096,0.2482,0.4988,0.3021,0.2272,0.0422
13,FC Hradec Králové,2024/25,FC Hradec Králové 2024/25,1,Main body of the league,0.604,1.4086,-1.507,0.6079,34,424,40,0.1211,0.0779,1.5097,-0.3333,16.2661,0.5993,0.2382,0.3184,12.4706,0.3396,0.25,0.0943,0.2264,0.4646,0.3656,0.3019,0.059
14,FC Slovan Liberec,2022/23,FC Slovan Liberec 2022/23,1,Main body of the league,0.604,-0.0613,1.0754,0.99,34,416,50,0.1064,0.0552,1.3015,0.1691,18.6348,0.5977,0.1851,0.4495,12.2353,0.3822,0.262,0.1202,0.1514,0.5385,0.2861,0.1803,0.0288
14,FC Slovan Liberec,2023/24,FC Slovan Liberec 2023/24,1,Main body of the league,0.604,-1.4938,1.3768,-0.4564,32,439,44,0.1034,0.052,1.4183,-0.0433,19.056,0.5897,0.1708,0.4715,13.7188,0.3371,0.2597,0.1002,0.1458,0.5604,0.2688,0.1868,0.0228
14,FC Slovan Liberec,2024/25,FC Slovan Liberec 2024/25,1,Main body of the league,0.604,0.1318,1.6413,0.0115,32,425,46,0.1216,0.0734,1.6156,-0.1781,17.758,0.5847,0.2212,0.3835,13.2812,0.3694,0.2706,0.1082,0.1835,0.4965,0.3176,0.2518,0.0753
17,SK Dynamo České Budějovice,2022/23,SK Dynamo České Budějovice 2022/23,1,Main body of the league,0.604,-3.026,0.6021,-0.2226,32,318,35,0.0881,0.0502,0.8753,0.2185,19.3073,0.6175,0.1447,0.4937,9.9375,0.3333,0.2579,0.1101,0.2264,0.5849,0.2201,0.1981,0.0126
17,SK Dynamo České Budějovice,2023/24,SK Dynamo České Budějovice 2023/24,1,Main body of the league,0.604,-3.5126,1.3872,1.3712,35,434,40,0.0913,0.045,1.1317,0.0111,19.1675,0.6174,0.1475,0.5415,12.4,0.3226,0.265,0.0922,0.1359,0.5207,0.2995,0.1544,0.0276
17,SK Dynamo České Budějovice,2024/25,SK Dynamo České Budějovice 2024/25,1,Main body of the league,0.604,-4.6767,-2.9353,1.2767,35,349,15,0.099,0.0574,0.987,-0.5584,17.5328,0.5974,0.1662,0.4298,9.9714,0.2693,0.3095,0.043,0.2034,0.4097,0.4126,0.255,0.0516
18,Mladá Boleslav,2022/23,Mladá Boleslav 2022/23,1,Main body of the league,0.604,-3.1155,-0.7624,0.7715,32,432,39,0.0851,0.0532,1.149,0.0697,18.9304,0.5842,0.125,0.4769,13.5,0.3426,0.2963,0.0903,0.1944,0.5162,0.3264,0.1505,0.0255
18,Mladá Boleslav,2023/24,Mladá Boleslav 2023/24,1,Main body of the league,0.604,0.2226,0.8878,1.2536,36,461,52,0.1069,0.0633,1.369,0.0754,18.0835,0.6228,0.1974,0.3991,12.8056,0.3341,0.295,0.1128,0.1584,0.5336,0.2885,0.2061,0.0412
18,Mladá Boleslav,2024/25,Mladá Boleslav 2024/25,1,Main body of the league,0.604,-1.1128,2.2311,0.0237,35,431,46,0.1181,0.0709,1.4541,-0.1398,17.6454,0.607,0.2204,0.3689,12.3143,0.3063,0.3039,0.1067,0.1206,0.4826,0.3016,0.232,0.0719
19,MFK Karviná,2023/24,MFK Karviná 2023/24,1,Main body of the league,0.604,-2.6251,-0.5196,-1.2298,35,423,35,0.0954,0.0544,1.1531,-0.1531,18.8233,0.5496,0.1324,0.4634,12.0857,0.3073,0.2553,0.0827,0.1939,0.5366,0.2388,0.1891,0.0355
19,MFK Karviná,2024/25,MFK Karviná 2024/25,1,Main body of the league,0.604,-1.7405,3.5243,-1.0813,31,384,39,0.1135,0.0663,1.4064,-0.1483,18.1331,0.5845,0.2135,0.4193,12.3871,0.3464,0.2448,0.1016,0.1484,0.5312,0.263,0.1901,0.0729
21,Dukla Praha,2024/25,Dukla Praha 2024/25,1,Main body of the league,0.604,-3.4506,-0.4074,1.1956,35,356,34,0.1066,0.0655,1.084,-0.1126,17.7594,0.5355,0.1657,0.4045,10.1714,0.2893,0.2865,0.0955,0.1573,0.4242,0.3764,0.2163,0.0562
```

### `tables/table4d_pca_loadings.csv`

```csv
Feature,PC1 chance creation (42.9%),PC2 finishing (16.5%),PC3 chance source (11.1%)
mean_shot_quality,0.348,0.017,0.011
median_shot_quality,0.336,-0.058,-0.075
xg_per_match,0.333,0.015,-0.147
goals_minus_xg_per_match,0.124,0.352,0.341
mean_distance_m,-0.311,0.251,-0.069
mean_angle_proxy,0.075,0.26,0.3
high_quality_share,0.34,0.026,-0.066
low_quality_share,-0.33,0.103,0.028
shots_per_match,0.271,0.025,-0.231
on_target_rate,0.211,0.358,0.214
blocked_rate,-0.14,-0.082,-0.035
goal_rate,0.294,0.256,0.217
header_share,0.131,-0.45,-0.002
assisted_share,0.041,0.341,-0.52
set_piece_share,-0.006,-0.318,0.552
close_range_share,0.251,-0.325,-0.095
fast_break_share,0.084,0.055,0.157
```

### `tables/table4e_feature_redundancy.csv`

```csv
Feature A,Feature B,Pearson r
median_shot_quality,low_quality_share,-0.969
mean_shot_quality,high_quality_share,0.95
xg_per_match,shots_per_match,0.905
mean_distance_m,low_quality_share,0.902
assisted_share,set_piece_share,-0.888
mean_shot_quality,median_shot_quality,0.884
median_shot_quality,mean_distance_m,-0.879
mean_shot_quality,low_quality_share,-0.869
median_shot_quality,high_quality_share,0.866
mean_shot_quality,xg_per_match,0.844
mean_distance_m,close_range_share,-0.84
xg_per_match,high_quality_share,0.838
high_quality_share,low_quality_share,-0.834
mean_shot_quality,mean_distance_m,-0.814
mean_distance_m,high_quality_share,-0.804
```

### `tables/table4f_gap_statistic.csv`

```csv
Feature set,k,log_Wk,gap,s_k
Full (17 features),1,6.5307,0.463,0.0652
Full (17 features),2,6.0678,0.4777,0.0599
Full (17 features),3,5.8628,0.4873,0.0608
Full (17 features),4,5.7058,0.4786,0.0623
Full (17 features),5,5.5915,0.4676,0.0744
Full (17 features),6,5.4784,0.4584,0.0699
Full (17 features),7,5.3668,0.4511,0.0771
Reduced (9 features),1,5.8929,0.532,0.0597
Reduced (9 features),2,5.6151,0.4426,0.0599
Reduced (9 features),3,5.4401,0.4183,0.0617
Reduced (9 features),4,5.2889,0.4045,0.0618
Reduced (9 features),5,5.155,0.4125,0.0757
Reduced (9 features),6,5.0127,0.4361,0.0715
Reduced (9 features),7,4.8657,0.465,0.0772
```

### `tables/table4g_pca_scree.csv`

```csv
Feature set,Component,Variance explained (%),Cumulative variance (%),Eigenvalue
Full (17 features),PC1,42.9,42.9,7.44
Full (17 features),PC2,16.5,59.4,2.87
Full (17 features),PC3,11.1,70.5,1.92
Full (17 features),PC4,8.3,78.7,1.44
Full (17 features),PC5,5.3,84.1,0.92
Full (17 features),PC6,4.5,88.6,0.79
Full (17 features),PC7,3.8,92.4,0.66
Full (17 features),PC8,2.4,94.8,0.41
Full (17 features),PC9,1.9,96.6,0.32
Full (17 features),PC10,1.3,97.9,0.22
Full (17 features),PC11,0.8,98.7,0.14
Full (17 features),PC12,0.5,99.2,0.09
Full (17 features),PC13,0.3,99.5,0.05
Full (17 features),PC14,0.3,99.8,0.05
Full (17 features),PC15,0.1,99.9,0.02
Full (17 features),PC16,0.1,100.0,0.02
Full (17 features),PC17,0.0,100.0,0.0
Reduced (9 features),PC1,30.2,30.2,2.78
Reduced (9 features),PC2,18.6,48.9,1.71
Reduced (9 features),PC3,14.6,63.4,1.34
Reduced (9 features),PC4,12.8,76.2,1.18
Reduced (9 features),PC5,7.7,83.9,0.71
Reduced (9 features),PC6,7.3,91.2,0.67
Reduced (9 features),PC7,3.8,95.1,0.35
Reduced (9 features),PC8,2.9,97.9,0.26
Reduced (9 features),PC9,2.1,100.0,0.19
```

### `tables/table4h_component_invariance.csv`

```csv
Components A,Components B,Adjusted Rand index
2,3,1.0
2,4,1.0
3,4,1.0
```

### `tables/table5_transition_matrix.csv`

```csv
From cluster,To cluster 0,To cluster 1,P(to 0),P(to 1)
Cluster 0,5,1,0.833,0.167
Cluster 1,0,24,0.0,1.0
Summary,,,retention 0.967,Cramer's V 0.890
```

### `tables/table5b_profile_repeatability.csv`

```csv
Feature,n team-season pairs,Year-to-year r,95% CI,p,ICC (team)
chance_creation_axis,30,0.764,"[0.56, 0.88]",0.0,0.78
finishing_axis,30,0.232,"[-0.14, 0.55]",0.2177,0.205
chance_source_axis,30,0.383,"[0.03, 0.65]",0.0367,0.331
mean_shot_quality,30,0.673,"[0.41, 0.83]",0.0,0.273
median_shot_quality,30,0.71,"[0.47, 0.85]",0.0,0.234
xg_per_match,30,0.885,"[0.77, 0.94]",0.0,0.726
goals_minus_xg_per_match,30,0.38,"[0.02, 0.65]",0.0385,-0.243
mean_distance_m,30,0.658,"[0.39, 0.82]",0.0001,0.358
mean_angle_proxy,30,0.186,"[-0.19, 0.51]",0.3252,0.036
high_quality_share,30,0.707,"[0.47, 0.85]",0.0,0.248
low_quality_share,30,0.681,"[0.43, 0.84]",0.0,0.357
shots_per_match,30,0.776,"[0.58, 0.89]",0.0,0.759
on_target_rate,30,0.294,"[-0.07, 0.59]",0.1145,0.247
blocked_rate,30,0.55,"[0.24, 0.76]",0.0016,0.231
goal_rate,30,0.407,"[0.05, 0.67]",0.0256,0.306
header_share,30,0.282,"[-0.09, 0.58]",0.1313,0.294
assisted_share,30,0.451,"[0.11, 0.70]",0.0124,0.006
set_piece_share,30,0.331,"[-0.03, 0.62]",0.0735,-0.057
close_range_share,30,0.435,"[0.09, 0.69]",0.0162,0.023
fast_break_share,30,0.589,"[0.29, 0.78]",0.0006,-0.358
```

### `tables/table5c_trajectory_typology.csv`

```csv
Team,Seasons,Chance-creation axis by season,Clusters,Range on chance-creation axis,Mean short-rest shot-quality delta,Trajectory
1. FC Slovácko,3,"-0.70, -0.96, -1.47",1->1->1,0.77,-0.019,Stable profile (load-sensitive)
AC Sparta Praha,3,"+4.91, +5.61, +0.84",0->0->1,4.77,-0.0006,Oscillating profile
Baník Ostrava,3,"-0.10, -2.03, +1.53",1->1->1,3.57,-0.033,Oscillating profile (load-sensitive)
Bohemians Praha 1905,3,"+0.41, -1.40, -0.26",1->1->1,1.81,-0.0081,Oscillating profile
FC Hradec Králové,3,"-0.86, +0.48, +1.41",1->1->1,2.26,-0.0261,Progressive improvement (load-sensitive)
FC Slovan Liberec,3,"-0.06, -1.49, +0.13",1->1->1,1.63,0.0262,Oscillating profile
FC Viktoria Plzeň,3,"+3.53, +5.74, +4.70",0->0->0,2.22,-0.01,Oscillating profile
FK Jablonec,3,"-1.25, -0.04, +1.93",1->1->1,3.18,0.0262,Progressive improvement
FK Pardubice,3,"-2.51, -0.51, -2.75",1->1->1,2.25,0.0188,Oscillating profile
FK Teplice,3,"-0.98, -3.50, -1.83",1->1->1,2.52,-0.0103,Oscillating profile
Mladá Boleslav,3,"-3.12, +0.22, -1.11",1->1->1,3.34,0.005,Oscillating profile
SK Dynamo České Budějovice,3,"-3.03, -3.51, -4.68",1->1->1,1.65,-0.0016,Oscillating profile
SK Sigma Olomouc,3,"-0.11, +0.48, +0.14",1->1->1,0.59,-0.0087,Stable profile
SK Slavia Praha,3,"+5.80, +4.04, +6.62",0->0->0,2.58,-0.0253,Oscillating profile (load-sensitive)
```

### `tables/table6_load_effects.csv`

```csv
Exposure,Outcome,Family,Estimate,CI low,CI high,p,N,Model,Scale,q (BH-FDR)
Short rest (<= 3 days vs >= 4),Shot quality (logit),primary,-0.0195,-0.0761,0.0372,0.5003,20782,LMM (team random intercept),units of outcome,
Short rest (<= 3 days vs >= 4),Shot distance (m),secondary,0.0838,-0.3717,0.5393,0.7184,20782,LMM (team random intercept),units of outcome,0.8875
Short rest (<= 3 days vs >= 4),Shot angle proxy (rad),secondary,-0.0051,-0.0254,0.0151,0.6185,20782,LMM (team random intercept),units of outcome,0.866
Short rest (<= 3 days vs >= 4),Shot on target,secondary,0.9898,0.88,1.1132,0.8637,20782,GEE logistic (clustered on match),odds ratio,0.916
Short rest (<= 3 days vs >= 4),Shot blocked,secondary,0.9734,0.8465,1.1194,0.7057,20782,GEE logistic (clustered on match),odds ratio,0.8875
Short rest (<= 3 days vs >= 4),Goal,secondary,0.9506,0.7831,1.154,0.6085,20782,GEE logistic (clustered on match),odds ratio,0.866
Short rest (<= 3 days vs >= 4),High-quality shot (SQ >= 0.15),secondary,0.9276,0.7872,1.0929,0.3689,20782,GEE logistic (clustered on match),odds ratio,0.7042
Short rest (<= 3 days vs >= 4),Low-quality shot (SQ < 0.05),secondary,1.0077,0.8899,1.1412,0.9036,20782,GEE logistic (clustered on match),odds ratio,0.916
Matches in previous 14 days (per match),Shot quality (logit),primary,0.0005,-0.0177,0.0187,0.956,20782,LMM (team random intercept),units of outcome,
Matches in previous 14 days (per match),Shot distance (m),secondary,0.1394,-0.0067,0.2856,0.0615,20782,LMM (team random intercept),units of outcome,0.2584
Matches in previous 14 days (per match),Shot angle proxy (rad),secondary,-0.0032,-0.0095,0.0032,0.3329,20782,LMM (team random intercept),units of outcome,0.7042
Matches in previous 14 days (per match),Shot on target,secondary,1.0026,0.9677,1.0387,0.8872,20782,GEE logistic (clustered on match),odds ratio,0.916
Matches in previous 14 days (per match),Shot blocked,secondary,0.9871,0.9467,1.0293,0.544,20782,GEE logistic (clustered on match),odds ratio,0.866
Matches in previous 14 days (per match),Goal,secondary,1.0789,1.0188,1.1425,0.0094,20782,GEE logistic (clustered on match),odds ratio,0.0656
Matches in previous 14 days (per match),High-quality shot (SQ >= 0.15),secondary,1.0759,1.0219,1.1327,0.0053,20782,GEE logistic (clustered on match),odds ratio,0.0561
Matches in previous 14 days (per match),Low-quality shot (SQ < 0.05),secondary,0.9706,0.9313,1.0115,0.1564,20782,GEE logistic (clustered on match),odds ratio,0.4737
"Days since previous fixture (per day, capped at 14)",Shot quality (logit),primary,0.0052,-0.0008,0.0111,0.0907,20782,LMM (team random intercept),units of outcome,
"Days since previous fixture (per day, capped at 14)",Shot distance (m),secondary,-0.077,-0.1251,-0.029,0.0017,20782,LMM (team random intercept),units of outcome,0.0352
"Days since previous fixture (per day, capped at 14)",Shot angle proxy (rad),secondary,0.0015,-0.0006,0.0036,0.1579,20782,LMM (team random intercept),units of outcome,0.4737
"Days since previous fixture (per day, capped at 14)",Shot on target,secondary,1.015,1.0027,1.0274,0.0166,20782,GEE logistic (clustered on match),odds ratio,0.0869
"Days since previous fixture (per day, capped at 14)",Shot blocked,secondary,0.9925,0.9785,1.0068,0.3028,20782,GEE logistic (clustered on match),odds ratio,0.7042
"Days since previous fixture (per day, capped at 14)",Goal,secondary,1.0011,0.9803,1.0224,0.916,20782,GEE logistic (clustered on match),odds ratio,0.916
"Days since previous fixture (per day, capped at 14)",High-quality shot (SQ >= 0.15),secondary,0.9912,0.9727,1.0102,0.3627,20782,GEE logistic (clustered on match),odds ratio,0.7042
"Days since previous fixture (per day, capped at 14)",Low-quality shot (SQ < 0.05),secondary,1.0057,0.9922,1.0195,0.4077,20782,GEE logistic (clustered on match),odds ratio,0.7135
```

### `tables/table6b_calendar_definitions.csv`

```csv
Calendar definition,Short-rest shots,Estimate,CI low,CI high,p,N,Model,Scale
League-only calendar,693,0.0026,-0.0725,0.0777,0.9463,20719,LMM (team random intercept),units of outcome
All-competition calendar,1260,-0.0195,-0.0761,0.0372,0.5003,20782,LMM (team random intercept),units of outcome
"All-competition, alternative cut-points",4326,0.0039,-0.0307,0.0385,0.8266,20782,LMM (team random intercept),units of outcome
```

### `tables/table6c_european_load.csv`

```csv
Exposure,Exposed shots,Estimate,CI low,CI high,p,N,Model,Scale
European match in previous 7 days,2022,0.0099,-0.0445,0.0643,0.7206,6760,LMM (team random intercept),units of outcome
Cup match in previous 7 days,1624,-0.0582,-0.1086,-0.0079,0.0233,20782,LMM (team random intercept),units of outcome
Upcoming cup/European fixture within 3 days,614,-0.0237,-0.1032,0.0559,0.5597,20782,LMM (team random intercept),units of outcome
European participant (team-season),6760,0.0799,0.0307,0.1291,0.0015,20782,LMM (team random intercept),units of outcome
```

### `tables/table6d_team_resilience.csv`

```csv
Team,delta,lo,hi,n_short
Baník Ostrava,-0.0271,-0.0457,-0.0074,71
SK Slavia Praha,-0.0267,-0.0492,-0.0003,87
FC Hradec Králové,-0.0203,-0.042,0.0049,63
SK Sigma Olomouc,-0.0181,-0.0387,0.0059,67
FK Teplice,-0.0096,-0.0303,0.0159,79
AC Sparta Praha,-0.0096,-0.0324,0.0146,94
Bohemians Praha 1905,-0.0082,-0.0296,0.0164,95
1. FC Slovácko,0.0034,-0.0243,0.0364,86
SK Dynamo České Budějovice,0.0075,-0.0164,0.0339,52
Mladá Boleslav,0.01,-0.0138,0.0366,116
FK Pardubice,0.0133,-0.0132,0.0445,54
FC Viktoria Plzeň,0.018,-0.0112,0.0481,108
FK Jablonec,0.0198,-0.0085,0.0514,99
FC Slovan Liberec,0.0218,-0.0159,0.0646,80
```

### `tables/table7_robustness.csv`

```csv
Analysis,Variation,Metric,Value,Primary-analysis value,Note
Shot-quality model,Primary specification,AUC (test season),0.7747,0.7747,
Shot-quality model,Primary specification,Brier skill score,0.1461,0.1461,
Shot-quality model,Excluding penalties,AUC (test season),0.7591,0.7747,"n = 7,033"
Shot-quality model,Excluding penalties,Brier skill score,0.1065,0.1461,
Shot-quality model,Excluding added-time shots,AUC (test season),0.7729,0.7747,"n = 6,456"
Shot-quality model,Excluding added-time shots,Brier skill score,0.1425,0.1461,
Shot-quality model,Excluding set-piece shots,AUC (test season),0.7561,0.7747,"n = 4,779"
Shot-quality model,Excluding set-piece shots,Brier skill score,0.1116,0.1461,
Shot-quality model,Open play only,AUC (test season),0.7561,0.7747,"n = 4,779"
Shot-quality model,Open play only,Brier skill score,0.1116,0.1461,
Shot-quality model,Metric-corrected distance/angle + visible goal angle,AUC (test season),0.7751,0.7747,Replaces the normalised proxies pre-specified in 5.2
Shot-quality model,Excluding the `situation` predictor,AUC (test season),0.7751,0.7747,`situation` taxonomy changes at the 2024/25 boundary (step 1b)
Shot-quality model,Excluding the `situation` predictor,Calibration intercept,-0.2755,-0.3199,
Shot-quality model,Calibration in 2022/23,Calibration slope,1.0379,1.0,
Shot-quality model,Calibration in 2022/23,Calibration intercept,0.1554,0.0,
Shot-quality model,Calibration in 2022/23,Observed / predicted goals,1.1226,1.0,
Shot-quality model,Calibration in 2023/24,Calibration slope,1.0597,1.0,
Shot-quality model,Calibration in 2023/24,Calibration intercept,-0.0274,0.0,
Shot-quality model,Calibration in 2023/24,Observed / predicted goals,0.9798,1.0,
Shot-quality model,Calibration in 2024/25,Calibration slope,0.9823,1.0,
Shot-quality model,Calibration in 2024/25,Calibration intercept,-0.2373,0.0,
Shot-quality model,Calibration in 2024/25,Observed / predicted goals,0.8392,1.0,
Shot-quality model,Train 2022/23 -> test 2023/24,AUC,0.7885,0.7747,
Shot-quality model,Train 2023/24 -> test 2024/25,AUC,0.7776,0.7747,
Clustering,All profile features (primary),Silhouette (k = 2),0.4176,0.4176,17 features
Clustering,All profile features (primary),Minimum bootstrap Jaccard (k = 2),0.6003,0.6003,0.60 is the floor for a reportable cluster
Clustering,Spatial features only,Silhouette (k = 2),0.3925,0.4176,6 features
Clustering,Spatial features only,Minimum bootstrap Jaccard (k = 2),0.5381,0.6003,0.60 is the floor for a reportable cluster
Clustering,Shot-quality features only,Silhouette (k = 2),0.5651,0.4176,8 features
Clustering,Shot-quality features only,Minimum bootstrap Jaccard (k = 2),0.6207,0.6003,0.60 is the floor for a reportable cluster
Clustering,Shot-selection features only,Silhouette (k = 2),0.206,0.4176,9 features
Clustering,Shot-selection features only,Minimum bootstrap Jaccard (k = 2),0.4361,0.6003,0.60 is the floor for a reportable cluster
Clustering,Profile + competitive-load response features,Silhouette (k = 2),0.4292,0.4176,19 features
Clustering,Profile + competitive-load response features,Minimum bootstrap Jaccard (k = 2),0.6061,0.6003,0.60 is the floor for a reportable cluster
Clustering,Within-season standardisation (primary),Minimum bootstrap Jaccard (k = 2),0.6003,0.6003,
Clustering,Pooled standardisation,Minimum bootstrap Jaccard (k = 2),0.5426,0.6003,
Profile construction,No-situation workflow,Team-seasons retained,48.0,48.0,OOF shot quality refit without `situation`; situation-derived profile features excluded below
Profile construction,No-situation workflow,mean_shot_quality correlation,0.9932,1.0,Compared with the primary team-season profile values
Profile construction,No-situation workflow,mean_shot_quality mean absolute difference,0.0014,0.0,
Profile construction,No-situation workflow,xg_per_match correlation,0.9983,1.0,Compared with the primary team-season profile values
Profile construction,No-situation workflow,xg_per_match mean absolute difference,0.0184,0.0,
Profile construction,No-situation workflow,high_quality_share correlation,0.9805,1.0,Compared with the primary team-season profile values
Profile construction,No-situation workflow,high_quality_share mean absolute difference,0.0082,0.0,
Profile construction,No-situation workflow,Cluster adjusted Rand index,1.0,1.0,Primary and no-situation workflows both exclude situation-derived profile features
Profile construction,No-situation workflow,Silhouette (k = 2),0.4854,0.4866,
Profile construction,No-situation workflow,Minimum bootstrap Jaccard (k = 2),0.5908,0.6,0.60 is the floor for a reportable cluster
Competitive load,Primary specification,Short-rest effect on logit shot quality,-0.0195,-0.0195,p = 0.500
Competitive load,Alternative rest cut-points (<=4 / 5-7 / >=8 days),Short-rest effect on logit shot quality,0.0039,-0.0195,"n = 20,782, p = 0.827"
Competitive load,League-only calendar,Short-rest effect on logit shot quality,0.0022,-0.0195,"n = 20,782, p = 0.954"
Competitive load,Excluding post-break fixtures,Short-rest effect on logit shot quality,-0.0195,-0.0195,"n = 19,585, p = 0.501"
Competitive load,Excluding penalties,Short-rest effect on logit shot quality,-0.0212,-0.0195,"n = 20,564, p = 0.430"
Competitive load,Open play only,Short-rest effect on logit shot quality,-0.0172,-0.0195,"n = 14,750, p = 0.599"
Competitive load,European participants only,Short-rest effect on logit shot quality,-0.0816,-0.0195,"n = 6,760, p = 0.107"
Competitive load,Non-European participants only,Short-rest effect on logit shot quality,0.0115,-0.0195,"n = 14,022, p = 0.745"
Competitive load,Adjusting for cup and European exposure,Short-rest effect on logit shot quality,0.0085,-0.0195,p = 0.787
Competitive load,Team and opponent fixed effects (cluster-robust SE),Short-rest effect on logit shot quality,-0.0262,-0.0195,p = 0.392
Competitive load,No team identifiers,Short-rest effect on logit shot quality,-0.0177,-0.0195,p = 0.565
Competitive load,Season 2022/23 only,Short-rest effect on logit shot quality,-0.1786,-0.0195,p = 0.005
Competitive load,Season 2023/24 only,Short-rest effect on logit shot quality,0.0335,-0.0195,p = 0.479
Competitive load,Season 2024/25 only,Short-rest effect on logit shot quality,0.0057,-0.0195,p = 0.901
Player profiles,Minimum 25 shots per player-season,Player-seasons retained,303.0,303.0,
Player profiles,Minimum 25 shots per player-season,SD of finishing over expectation,2.1438,,Reported as threshold sensitivity; retained-player composition changes with the cutoff
Player profiles,Minimum 30 shots per player-season,Player-seasons retained,234.0,303.0,
Player profiles,Minimum 30 shots per player-season,SD of finishing over expectation,2.3138,,Reported as threshold sensitivity; retained-player composition changes with the cutoff
Player profiles,Minimum 40 shots per player-season,Player-seasons retained,124.0,303.0,
Player profiles,Minimum 40 shots per player-season,SD of finishing over expectation,2.589,,Reported as threshold sensitivity; retained-player composition changes with the cutoff
Player profiles,Minimum 50 shots per player-season,Player-seasons retained,70.0,303.0,
Player profiles,Minimum 50 shots per player-season,SD of finishing over expectation,2.8765,,Reported as threshold sensitivity; retained-player composition changes with the cutoff
```

### `tables/table7b_no_situation_profile_comparison.csv`

```csv
Feature,Primary mean,No-situation mean,Pearson r,Mean absolute difference,Maximum absolute difference
mean_shot_quality,0.1085,0.1086,0.9932,0.0014,0.0041
median_shot_quality,0.0654,0.0662,0.9859,0.0015,0.0051
xg_per_match,1.4148,1.4156,0.9983,0.0184,0.0669
goals_minus_xg_per_match,-0.0405,-0.0413,0.9965,0.0184,0.0669
mean_distance_m,17.7911,17.7911,1.0,0.0,0.0
mean_angle_proxy,0.5889,0.5889,1.0,0.0,0.0
high_quality_share,0.1879,0.1928,0.9805,0.0082,0.0329
low_quality_share,0.3955,0.3945,0.9854,0.0093,0.0303
shots_per_match,12.8903,12.8903,1.0,0.0,0.0
on_target_rate,0.3371,0.3371,1.0,0.0,0.0
blocked_rate,0.26,0.26,1.0,0.0,0.0
goal_rate,0.1051,0.1051,1.0,0.0,0.0
header_share,0.1879,0.1879,1.0,0.0,0.0
close_range_share,0.222,0.222,1.0,0.0,0.0
```

### `tables/table8_split_half_reliability.csv`

```csv
Feature,Pearson r,Pearson 95% bootstrap CI,Spearman rho,Spearman 95% bootstrap CI,MAE first vs second half,n team-seasons
shots_per_match,0.637,"[0.430, 0.775]",0.541,"[0.254, 0.731]",1.651,48
xg_per_match,0.751,"[0.572, 0.868]",0.676,"[0.453, 0.812]",0.243,48
mean_shot_quality,0.593,"[0.355, 0.757]",0.571,"[0.310, 0.745]",0.013,48
high_quality_share,0.602,"[0.406, 0.765]",0.561,"[0.331, 0.741]",0.04,48
chance_creation_axis,0.628,"[0.418, 0.773]",0.506,"[0.232, 0.716]",2.209,48
goals_per_match,0.626,"[0.412, 0.779]",0.526,"[0.247, 0.737]",0.34,48
goals_minus_xg_per_match,0.452,"[0.124, 0.665]",0.322,"[-0.005, 0.600]",0.278,48
on_target_rate,0.381,"[0.143, 0.594]",0.375,"[0.108, 0.616]",0.035,48
finishing_axis,0.552,"[0.341, 0.711]",0.557,"[0.310, 0.730]",1.641,48
```

### `tables/table8b_split_half_differences.csv`

```csv
Comparison,Pearson r difference,95% paired cluster bootstrap CI,Two-sided paired bootstrap p
chance_creation_axis minus finishing_axis,0.076,"[-0.354, 0.282]",0.663
xg_per_match minus goals_minus_xg_per_match,0.299,"[-0.039, 0.618]",0.067
```

### `tables/table9_practical_validity.csv`

```csv
Outcome,Term,Estimate,SE cluster(team),p cluster(team),Team-cluster bootstrap 95% CI,Team-cluster bootstrap p,Model R2,Inference note
points_per_match,process_chance_creation_axis,0.1732,0.0157,< .001,"[0.132, 0.201]",< .001,0.815,Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
points_per_match,finishing_axis,0.1076,0.0213,< .001,"[0.060, 0.149]",< .001,0.815,Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
goal_difference_per_match,process_chance_creation_axis,0.2745,0.0253,< .001,"[0.212, 0.317]",< .001,0.845,Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
goal_difference_per_match,finishing_axis,0.1814,0.035,< .001,"[0.100, 0.245]",< .001,0.845,Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
nonpen_goal_difference_per_match,process_chance_creation_axis,0.2501,0.0197,< .001,"[0.198, 0.284]",< .001,0.836,Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
nonpen_goal_difference_per_match,finishing_axis,0.1582,0.0328,< .001,"[0.083, 0.216]",< .001,0.836,Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
points_per_match,Spearman(process_chance_creation_axis),0.6662,,,,,,Descriptive rank association
goal_difference_per_match,Spearman(process_chance_creation_axis),0.712,,,,,,Descriptive rank association
points_per_match,Spearman(finishing_axis),0.1024,,,,,,Descriptive rank association
goal_difference_per_match,Spearman(finishing_axis),0.0973,,,,,,Descriptive rank association
```

### `tables/table9b_opponent_adjusted_profiles.csv`

```csv
team_season_key,adjusted_xg_per_match,adjusted_shots_per_match,adjusted_mean_shot_quality,team_id,season_name,team_name,chance_creation_axis,xg_per_match,raw_xg_rank,adjusted_xg_rank,rank_change_adjusted_minus_raw,Spearman adjusted xG vs chance axis
3_2022/23,1.5523,13.6238,0.1092,3,2022/23,AC Sparta Praha,4.9108,1.9798,1.0,1.0,0.0,0.721
4_2022/23,1.379,12.3855,0.1092,4,2022/23,SK Slavia Praha,5.8015,1.8234,2.0,2.0,0.0,0.721
1_2022/23,1.2915,13.0468,0.1008,1,2022/23,FC Viktoria Plzeň,3.5251,1.6991,3.0,3.0,0.0,0.721
10_2022/23,1.0098,10.7237,0.0942,10,2022/23,Baník Ostrava,-0.0976,1.4437,4.0,4.0,0.0,0.721
11_2022/23,0.8724,9.7694,0.094,11,2022/23,Bohemians Praha 1905,0.4098,1.2403,6.0,5.0,-1.0,0.721
2_2022/23,0.855,10.327,0.0851,2,2022/23,SK Sigma Olomouc,-0.1091,1.2356,7.0,6.0,-1.0,0.721
14_2022/23,0.8496,9.2855,0.0949,14,2022/23,FC Slovan Liberec,-0.0613,1.3015,5.0,7.0,2.0,0.721
18_2022/23,0.7387,10.7747,0.0756,18,2022/23,Mladá Boleslav,-3.1155,1.149,8.0,8.0,0.0,0.721
12_2022/23,0.7197,8.6885,0.0882,12,2022/23,1. FC Slovácko,-0.702,1.0977,13.0,9.0,-4.0,0.721
13_2022/23,0.7137,8.9705,0.0846,13,2022/23,FC Hradec Králové,-0.8563,1.1467,10.0,10.0,0.0,0.721
6_2022/23,0.7033,9.3067,0.0771,6,2022/23,FC Zlín,-0.6763,1.1238,12.0,11.0,-1.0,0.721
9_2022/23,0.7019,9.0409,0.0831,9,2022/23,FK Jablonec,-1.2491,1.1468,9.0,12.0,3.0,0.721
7_2022/23,0.6995,8.7665,0.0876,7,2022/23,FC Zbrojovka Brno,-1.2645,1.1239,11.0,13.0,2.0,0.721
8_2022/23,0.6361,8.9279,0.0749,8,2022/23,FK Pardubice,-2.5112,1.043,14.0,14.0,0.0,0.721
5_2022/23,0.609,7.2683,0.0909,5,2022/23,FK Teplice,-0.9784,1.0419,15.0,15.0,0.0,0.721
17_2022/23,0.4884,7.3548,0.0776,17,2022/23,SK Dynamo České Budějovice,-3.026,0.8753,16.0,16.0,0.0,0.721
4_2023/24,1.8876,15.8121,0.1127,4,2023/24,SK Slavia Praha,4.042,2.3126,1.0,1.0,0.0,0.721
1_2023/24,1.6463,13.0449,0.1159,1,2023/24,FC Viktoria Plzeň,5.742,2.0345,3.0,2.0,-1.0,0.721
3_2023/24,1.644,13.5361,0.1142,3,2023/24,AC Sparta Praha,5.6119,2.0522,2.0,3.0,1.0,0.721
10_2023/24,1.179,13.7955,0.0829,10,2023/24,Baník Ostrava,-2.0332,1.5436,4.0,4.0,0.0,0.721
18_2023/24,1.0196,10.713,0.0893,18,2023/24,Mladá Boleslav,0.2226,1.369,6.0,5.0,-1.0,0.721
14_2023/24,1.0131,11.2819,0.0919,14,2023/24,FC Slovan Liberec,-1.4938,1.4183,5.0,6.0,1.0,0.721
12_2023/24,0.9664,10.5437,0.0876,12,2023/24,1. FC Slovácko,-0.9646,1.325,8.0,7.0,-1.0,0.721
13_2023/24,0.931,9.7976,0.0981,13,2023/24,FC Hradec Králové,0.4801,1.3344,7.0,8.0,1.0,0.721
11_2023/24,0.8364,9.4946,0.1001,11,2023/24,Bohemians Praha 1905,-1.3972,1.2763,9.0,9.0,0.0,0.721
19_2023/24,0.7446,9.632,0.0821,19,2023/24,MFK Karviná,-2.6251,1.1531,11.0,10.0,-1.0,0.721
17_2023/24,0.7435,9.9773,0.0817,17,2023/24,SK Dynamo České Budějovice,-3.5126,1.1317,13.0,11.0,-2.0,0.721
5_2023/24,0.7401,10.3332,0.0823,5,2023/24,FK Teplice,-3.503,1.1379,12.0,12.0,0.0,0.721
9_2023/24,0.7224,8.439,0.0932,9,2023/24,FK Jablonec,-0.0438,1.16,10.0,13.0,3.0,0.721
8_2023/24,0.6987,8.1659,0.0912,8,2023/24,FK Pardubice,-0.5074,1.1225,14.0,14.0,0.0,0.721
6_2023/24,0.6786,8.6859,0.0843,6,2023/24,FC Zlín,-0.5014,1.1059,15.0,15.0,0.0,0.721
2_2023/24,0.6692,7.964,0.1021,2,2023/24,SK Sigma Olomouc,0.4838,1.0739,16.0,16.0,0.0,0.721
4_2024/25,2.157,15.668,0.1255,4,2024/25,SK Slavia Praha,6.6182,2.7152,1.0,1.0,0.0,0.721
1_2024/25,1.5204,12.4722,0.1182,1,2024/25,FC Viktoria Plzeň,4.7025,2.0418,2.0,2.0,0.0,0.721
10_2024/25,1.4386,13.8273,0.0986,10,2024/25,Baník Ostrava,1.5344,1.9409,3.0,3.0,0.0,0.721
3_2024/25,1.2671,14.0859,0.0866,3,2024/25,AC Sparta Praha,0.8383,1.7984,4.0,4.0,0.0,0.721
9_2024/25,1.2586,12.3203,0.0958,9,2024/25,FK Jablonec,1.9333,1.7519,5.0,5.0,0.0,0.721
14_2024/25,1.07,11.0183,0.0977,14,2024/25,FC Slovan Liberec,0.1318,1.6156,6.0,6.0,0.0,0.721
11_2024/25,1.0503,11.0044,0.0927,11,2024/25,Bohemians Praha 1905,-0.2644,1.588,7.0,7.0,0.0,0.721
13_2024/25,0.9421,10.0822,0.0979,13,2024/25,FC Hradec Králové,1.4086,1.5097,8.0,8.0,0.0,0.721
19_2024/25,0.8888,10.298,0.0862,19,2024/25,MFK Karviná,-1.7405,1.4064,10.0,9.0,-1.0,0.721
18_2024/25,0.8803,9.9178,0.0936,18,2024/25,Mladá Boleslav,-1.1128,1.4541,9.0,10.0,1.0,0.721
2_2024/25,0.8463,8.6667,0.1078,2,2024/25,SK Sigma Olomouc,0.1381,1.3406,11.0,11.0,0.0,0.721
5_2024/25,0.726,10.2465,0.0809,5,2024/25,FK Teplice,-1.8325,1.3047,12.0,12.0,0.0,0.721
12_2024/25,0.7136,8.384,0.092,12,2024/25,1. FC Slovácko,-1.4747,1.3025,13.0,13.0,0.0,0.721
21_2024/25,0.5304,7.8395,0.085,21,2024/25,Dukla Praha,-3.4506,1.084,14.0,14.0,0.0,0.721
8_2024/25,0.5029,7.3844,0.0858,8,2024/25,FK Pardubice,-2.7532,1.0485,15.0,15.0,0.0,0.721
17_2024/25,0.4598,7.8049,0.0745,17,2024/25,SK Dynamo České Budějovice,-4.6767,0.987,16.0,16.0,0.0,0.721
```

### `tables/tableS1_player_profiles_top.csv`

```csv
player_name,team_name,season_name,position,shots,goals,mean_shot_quality,total_xg,finishing_over_expectation,close_range_share,header_share,on_target_rate
Václav Jurečka,SK Slavia Praha,2022/23,F,49,20,0.199,9.727,10.273,0.265,0.102,0.653
Jan Kliment,SK Sigma Olomouc,2024/25,F,51,18,0.201,10.228,7.772,0.412,0.137,0.49
Ewerton,Baník Ostrava,2023/24,M,80,14,0.102,8.131,5.869,0.062,0.038,0.362
Veljko Birmančević,AC Sparta Praha,2023/24,F,70,16,0.146,10.222,5.778,0.171,0.086,0.486
Peter Olayinka,SK Slavia Praha,2022/23,M,39,11,0.134,5.223,5.777,0.359,0.282,0.436
Pavel Šulc,FC Viktoria Plzeň,2023/24,F,72,18,0.176,12.685,5.315,0.389,0.278,0.403
Marek Havlík,1. FC Slovácko,2023/24,M,75,12,0.091,6.796,5.204,0.093,0.04,0.36
Ondrej Lingr,SK Slavia Praha,2022/23,M,41,11,0.154,6.321,4.679,0.366,0.244,0.341
Lukáš Haraslín,AC Sparta Praha,2024/25,F,79,11,0.081,6.381,4.619,0.038,0.013,0.43
Roman Kvet,Bohemians Praha 1905,2022/23,M,49,11,0.134,6.565,4.435,0.184,0.082,0.51
Lukáš Haraslín,AC Sparta Praha,2023/24,F,72,12,0.105,7.572,4.428,0.042,0.014,0.431
Lamin Jawo,Mladá Boleslav,2023/24,F,32,7,0.09,2.885,4.115,0.219,0.281,0.406
Jan Chramosta,FK Jablonec,2022/23,F,77,15,0.142,10.915,4.085,0.13,0.013,0.442
Jakub Řezníček,FC Zbrojovka Brno,2022/23,F,95,19,0.157,14.938,4.062,0.305,0.242,0.495
Lukáš Juliš,SK Sigma Olomouc,2023/24,F,48,12,0.167,8.033,3.967,0.292,0.167,0.458
Tomáš Čvančara,AC Sparta Praha,2022/23,F,50,12,0.169,8.452,3.548,0.32,0.26,0.44
Matyáš Vojta,Mladá Boleslav,2024/25,F,35,10,0.187,6.538,3.462,0.514,0.229,0.6
Mick van Buren,FC Slovan Liberec,2022/23,F,68,14,0.158,10.777,3.223,0.235,0.235,0.471
Amar Memić,MFK Karviná,2024/25,M,63,10,0.108,6.808,3.192,0.143,0.063,0.365
Abdoull Tanko,Baník Ostrava,2023/24,F,51,9,0.114,5.836,3.164,0.216,0.216,0.451
```

### `tables/tableS2_stabilisation_curves.csv`

```csv
Matches,Feature,Pearson r with remainder,Spearman rho with remainder,n,Excluded team-seasons,Exclusion note
3,chance_creation_axis,0.293,0.265,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
3,finishing_axis,0.407,0.411,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
3,xg_per_match,0.573,0.574,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
3,shots_per_match,0.558,0.515,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
3,mean_shot_quality,0.336,0.315,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
3,goals_minus_xg_per_match,0.397,0.432,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
3,on_target_rate,0.279,0.299,47,1,FC Hradec Králové 2024/25 (early window 15 shots)
5,chance_creation_axis,0.568,0.49,48,0,
5,finishing_axis,0.478,0.506,48,0,
5,xg_per_match,0.702,0.711,48,0,
5,shots_per_match,0.612,0.571,48,0,
5,mean_shot_quality,0.48,0.458,48,0,
5,goals_minus_xg_per_match,0.599,0.552,48,0,
5,on_target_rate,0.296,0.307,48,0,
7,chance_creation_axis,0.62,0.566,48,0,
7,finishing_axis,0.533,0.54,48,0,
7,xg_per_match,0.719,0.715,48,0,
7,shots_per_match,0.669,0.614,48,0,
7,mean_shot_quality,0.497,0.465,48,0,
7,goals_minus_xg_per_match,0.556,0.492,48,0,
7,on_target_rate,0.248,0.198,48,0,
10,chance_creation_axis,0.688,0.628,48,0,
10,finishing_axis,0.618,0.602,48,0,
10,xg_per_match,0.768,0.723,48,0,
10,shots_per_match,0.707,0.649,48,0,
10,mean_shot_quality,0.578,0.574,48,0,
10,goals_minus_xg_per_match,0.432,0.374,48,0,
10,on_target_rate,0.415,0.406,48,0,
12,chance_creation_axis,0.708,0.659,48,0,
12,finishing_axis,0.605,0.644,48,0,
12,xg_per_match,0.79,0.718,48,0,
12,shots_per_match,0.652,0.578,48,0,
12,mean_shot_quality,0.598,0.57,48,0,
12,goals_minus_xg_per_match,0.486,0.455,48,0,
12,on_target_rate,0.357,0.33,48,0,
15,chance_creation_axis,0.74,0.673,48,0,
15,finishing_axis,0.625,0.634,48,0,
15,xg_per_match,0.773,0.679,48,0,
15,shots_per_match,0.629,0.561,48,0,
15,mean_shot_quality,0.667,0.618,48,0,
15,goals_minus_xg_per_match,0.513,0.408,48,0,
15,on_target_rate,0.42,0.455,48,0,
18,chance_creation_axis,0.686,0.637,48,0,
18,finishing_axis,0.621,0.613,48,0,
18,xg_per_match,0.751,0.673,48,0,
18,shots_per_match,0.667,0.567,48,0,
18,mean_shot_quality,0.645,0.604,48,0,
18,goals_minus_xg_per_match,0.459,0.317,48,0,
18,on_target_rate,0.363,0.397,48,0,
20,chance_creation_axis,0.703,0.646,48,0,
20,finishing_axis,0.553,0.523,48,0,
20,xg_per_match,0.747,0.644,48,0,
20,shots_per_match,0.683,0.585,48,0,
20,mean_shot_quality,0.652,0.592,48,0,
20,goals_minus_xg_per_match,0.563,0.405,48,0,
20,on_target_rate,0.373,0.372,48,0,
```

### `tables/tableS3_early_season_regression.csv`

```csv
Window,Target,Model,Holdout R2 (2024/25 remainder),R2 95% bootstrap CI,Holdout MAE,MAE 95% bootstrap CI,Holdout Spearman rho,Spearman 95% bootstrap CI,N train,N test
First 5 matches,chance creation axis,Naive early-value baseline,-1.387,"[-6.442, -0.012]",3.619,"[2.488, 4.828]",0.582,"[0.107, 0.856]",32,16
First 5 matches,chance creation axis,Ridge regression,-0.761,"[-2.297, -0.159]",3.013,"[2.071, 4.137]",0.597,"[0.102, 0.823]",32,16
First 5 matches,chance creation axis,Random forest sensitivity,-0.478,"[-2.653, 0.183]",2.667,"[1.718, 3.818]",0.524,"[-0.012, 0.828]",32,16
First 5 matches,finishing axis,Naive early-value baseline,-1.156,"[-4.107, 0.051]",1.79,"[1.139, 2.462]",0.444,"[-0.093, 0.801]",32,16
First 5 matches,finishing axis,Ridge regression,-2.315,"[-6.029, -1.093]",2.463,"[1.793, 3.153]",0.582,"[0.054, 0.887]",32,16
First 5 matches,finishing axis,Random forest sensitivity,-1.641,"[-4.640, -0.620]",2.151,"[1.486, 2.805]",0.274,"[-0.269, 0.732]",32,16
First 5 matches,xg per match,Naive early-value baseline,-0.015,"[-2.250, 0.479]",0.335,"[0.191, 0.498]",0.788,"[0.452, 0.922]",32,16
First 5 matches,xg per match,Ridge regression,0.316,"[-0.098, 0.834]",0.216,"[0.106, 0.388]",0.847,"[0.532, 0.956]",32,16
First 5 matches,xg per match,Random forest sensitivity,0.379,"[-0.068, 0.684]",0.238,"[0.127, 0.386]",0.694,"[0.257, 0.907]",32,16
First 5 matches,goals minus xg per match,Naive early-value baseline,-1.041,"[-3.512, -0.097]",0.245,"[0.171, 0.325]",0.326,"[-0.212, 0.734]",32,16
First 5 matches,goals minus xg per match,Ridge regression,-3.505,"[-7.049, -2.242]",0.385,"[0.293, 0.485]",0.303,"[-0.247, 0.754]",32,16
First 5 matches,goals minus xg per match,Random forest sensitivity,-3.537,"[-6.848, -2.361]",0.383,"[0.290, 0.488]",0.168,"[-0.430, 0.694]",32,16
First 10 matches,chance creation axis,Naive early-value baseline,-0.103,"[-3.044, 0.588]",2.417,"[1.548, 3.314]",0.612,"[0.205, 0.839]",32,16
First 10 matches,chance creation axis,Ridge regression,-0.54,"[-2.581, -0.011]",3.021,"[2.101, 3.944]",0.553,"[0.076, 0.841]",32,16
First 10 matches,chance creation axis,Random forest sensitivity,-0.396,"[-2.503, 0.193]",2.93,"[2.126, 3.806]",0.653,"[0.239, 0.889]",32,16
First 10 matches,finishing axis,Naive early-value baseline,-1.152,"[-5.819, 0.084]",1.877,"[1.366, 2.436]",0.594,"[0.211, 0.827]",32,16
First 10 matches,finishing axis,Ridge regression,-1.04,"[-4.319, -0.120]",1.873,"[1.407, 2.348]",0.676,"[0.293, 0.888]",32,16
First 10 matches,finishing axis,Random forest sensitivity,-1.669,"[-5.942, -0.370]",2.122,"[1.576, 2.704]",0.526,"[0.039, 0.801]",32,16
First 10 matches,xg per match,Naive early-value baseline,0.377,"[-1.223, 0.679]",0.303,"[0.198, 0.410]",0.726,"[0.358, 0.911]",32,16
First 10 matches,xg per match,Ridge regression,0.321,"[-0.095, 0.507]",0.282,"[0.168, 0.430]",0.6,"[0.161, 0.840]",32,16
First 10 matches,xg per match,Random forest sensitivity,0.393,"[-0.178, 0.625]",0.269,"[0.164, 0.405]",0.682,"[0.274, 0.910]",32,16
First 10 matches,goals minus xg per match,Naive early-value baseline,-0.961,"[-2.849, -0.338]",0.273,"[0.187, 0.371]",0.003,"[-0.540, 0.534]",32,16
First 10 matches,goals minus xg per match,Ridge regression,-2.209,"[-4.815, -1.315]",0.356,"[0.246, 0.474]",0.356,"[-0.237, 0.839]",32,16
First 10 matches,goals minus xg per match,Random forest sensitivity,-2.044,"[-4.380, -1.161]",0.331,"[0.219, 0.461]",-0.485,"[-0.872, 0.056]",32,16
First 30% of season,chance creation axis,Naive early-value baseline,-0.205,"[-3.038, 0.590]",2.542,"[1.615, 3.577]",0.521,"[0.058, 0.804]",32,16
First 30% of season,chance creation axis,Ridge regression,-0.537,"[-2.856, 0.125]",3.085,"[2.190, 3.991]",0.529,"[0.049, 0.809]",32,16
First 30% of season,chance creation axis,Random forest sensitivity,-0.327,"[-2.082, 0.277]",2.71,"[1.774, 3.746]",0.488,"[0.001, 0.805]",32,16
First 30% of season,finishing axis,Naive early-value baseline,-0.635,"[-4.491, 0.430]",1.465,"[0.951, 2.018]",0.75,"[0.427, 0.910]",32,16
First 30% of season,finishing axis,Ridge regression,-0.762,"[-4.244, 0.122]",1.682,"[1.245, 2.131]",0.735,"[0.414, 0.902]",32,16
First 30% of season,finishing axis,Random forest sensitivity,-1.096,"[-5.184, -0.019]",1.83,"[1.349, 2.328]",0.638,"[0.199, 0.898]",32,16
First 30% of season,xg per match,Naive early-value baseline,0.429,"[-0.820, 0.693]",0.299,"[0.203, 0.405]",0.726,"[0.324, 0.911]",32,16
First 30% of season,xg per match,Ridge regression,0.366,"[-0.065, 0.535]",0.276,"[0.159, 0.421]",0.629,"[0.204, 0.851]",32,16
First 30% of season,xg per match,Random forest sensitivity,0.333,"[-0.351, 0.505]",0.302,"[0.195, 0.435]",0.715,"[0.352, 0.884]",32,16
First 30% of season,goals minus xg per match,Naive early-value baseline,-0.874,"[-2.709, -0.281]",0.25,"[0.165, 0.348]",0.203,"[-0.313, 0.665]",32,16
First 30% of season,goals minus xg per match,Ridge regression,-2.297,"[-5.046, -1.392]",0.347,"[0.240, 0.463]",0.329,"[-0.274, 0.804]",32,16
First 30% of season,goals minus xg per match,Random forest sensitivity,-1.506,"[-3.468, -0.817]",0.278,"[0.176, 0.395]",-0.535,"[-0.870, -0.034]",32,16
```

## Model Output Files


### `models/shap_importance.csv`

```csv
feature,mean_abs_shap
distance_proxy,0.48008292485927817
body_part_head,0.19670258342736674
player_x,0.1869247067808405
centrality,0.09232376182216545
situation_fast_break,0.08035749085772101
lateral_offset,0.0795159820101467
score_diff_before,0.05061546766061681
situation_corner,0.04845142557522516
situation_penalty,0.04766099074210007
score_state_leading,0.03687874503266872
score_state_trailing,0.033113490819496806
inside_box,0.032193004125057406
player_y,0.031181622834790534
minute,0.023764520492360476
angle_proxy,0.022310992388301303
body_part_right_foot,0.011803821977352257
situation_open_play,0.011637313831584712
inside_six_yard,0.009738442905597326
body_part_left_foot,0.009663766572136082
central_corridor,0.009386748593966127
score_state_level,0.007227694333104301
second_half,0.006095446102336842
situation_assisted,0.003597656313375659
situation_free_kick,0.003544918677369721
situation_set_piece,0.0031940618557469973
situation_throw_in,0.0025938625336490383
late_game,0.001962411916543716
is_home_shot,0.0015735662561114927
added_time_flag,0.0004268961859911886
body_part_other,0.0003816663374238185
```

### `models/tuning_log.json`

```json
[
  {
    "family": "Regularised logistic regression",
    "config": "C=1.0",
    "inner_logloss": 0.27808904117085453,
    "inner_auc": 0.7827326869348479,
    "best_iter": null
  },
  {
    "family": "Regularised logistic regression",
    "config": "C=10.0",
    "inner_logloss": 0.27836945244894923,
    "inner_auc": 0.7822457525704575,
    "best_iter": null
  },
  {
    "family": "Regularised logistic regression",
    "config": "C=0.1",
    "inner_logloss": 0.27947114259422284,
    "inner_auc": 0.7814457458013243,
    "best_iter": null
  },
  {
    "family": "Random forest",
    "config": "depth=8,leaf=20",
    "inner_logloss": 0.2794451688921538,
    "inner_auc": 0.7772065762127828,
    "best_iter": null
  },
  {
    "family": "Random forest",
    "config": "depth=None,leaf=20",
    "inner_logloss": 0.27953347062692885,
    "inner_auc": 0.7768295395297226,
    "best_iter": null
  },
  {
    "family": "Random forest",
    "config": "depth=12,leaf=20",
    "inner_logloss": 0.2796094488728653,
    "inner_auc": 0.7766680675011504,
    "best_iter": null
  },
  {
    "family": "Random forest",
    "config": "depth=None,leaf=50",
    "inner_logloss": 0.28128821403722565,
    "inner_auc": 0.7752854317780176,
    "best_iter": null
  },
  {
    "family": "Random forest",
    "config": "depth=12,leaf=50",
    "inner_logloss": 0.28135969234977337,
    "inner_auc": 0.7750011684813047,
    "best_iter": null
  },
  {
    "family": "Random forest",
    "config": "depth=8,leaf=50",
    "inner_logloss": 0.281645557435797,
    "inner_auc": 0.7744508742254782,
    "best_iter": null
  },
  {
    "family": "XGBoost",
    "config": "depth=3,mcw=5",
    "inner_logloss": 0.27661022543907166,
    "inner_auc": 0.7829277024353568,
    "best_iter": 160
  },
  {
    "family": "XGBoost",
    "config": "depth=4,mcw=5",
    "inner_logloss": 0.27675655484199524,
    "inner_auc": 0.7818409140908369,
    "best_iter": 135
  },
  {
    "family": "XGBoost",
    "config": "depth=5,mcw=5",
    "inner_logloss": 0.2778375744819641,
    "inner_auc": 0.7794572122292448,
    "best_iter": 94
  },
  {
    "family": "XGBoost",
    "config": "depth=4,mcw=20",
    "inner_logloss": 0.27816957235336304,
    "inner_auc": 0.779860338279367,
    "best_iter": 152
  },
  {
    "family": "XGBoost",
    "config": "depth=3,mcw=20",
    "inner_logloss": 0.27847999334335327,
    "inner_auc": 0.7795142260584225,
    "best_iter": 179
  },
  {
    "family": "XGBoost",
    "config": "depth=5,mcw=20",
    "inner_logloss": 0.27873969078063965,
    "inner_auc": 0.7781501248663296,
    "best_iter": 125
  },
  {
    "family": "CatBoost",
    "config": "depth=4,l2=10",
    "inner_logloss": 0.27456623473219827,
    "inner_auc": 0.7879898454945377,
    "best_iter": 247
  },
  {
    "family": "CatBoost",
    "config": "depth=4,l2=3",
    "inner_logloss": 0.2753919026158092,
    "inner_auc": 0.7861836352984503,
    "best_iter": 340
  },
  {
    "family": "CatBoost",
    "config": "depth=6,l2=3",
    "inner_logloss": 0.27556119512038096,
    "inner_auc": 0.7849519955646062,
    "best_iter": 203
  },
  {
    "family": "CatBoost",
    "config": "depth=6,l2=10",
    "inner_logloss": 0.27594154302535306,
    "inner_auc": 0.7840866142810978,
    "best_iter": 202
  }
]
```

## Captured Statistical Software Logs

`s01_extract_audit.py` is an optional source-database extraction step and is not required to reproduce the analyses from the released processed data.

### `logs/s02_features.log`

```text

==============================================================================
STEP 2  Shot-level feature engineering
==============================================================================
  spatial + context features built for 21,311 shots
  calendar-load features built for 1,652 team-matches

  rest-category distribution of shots:
rest_category
long       0.589
normal     0.327
short      0.059
unknown    0.025

  goal rate by rest category:
                 mean   size
rest_category               
long           0.1049  12551
normal         0.1112   6971
short          0.1000   1260
unknown        0.1078    529

  missingness in key load features:
    days_since_any         2.482%
    days_since_league      2.778%
    matches_last_14        0.000%
  [data ] outputs/data/shots_features.parquet  (21,311 rows x 96 cols)
  [data ] outputs/data/team_match_load.parquet  (1,652 rows x 30 cols)
```

### `logs/s03_shot_quality.log`

```text

==============================================================================
STEP 3  Shot-quality model development and validation
==============================================================================
  train 14,209 shots (['2022/23', '2023/24']), test 7,102 shots (['2024/25']); training base rate 0.1095

  tuning Regularised logistic regression (3 configurations)
    C=0.1                inner log loss 0.27947   AUC 0.7814
    C=1.0                inner log loss 0.27809   AUC 0.7827
    C=10.0               inner log loss 0.27837   AUC 0.7822

  tuning Random forest (6 configurations)
    depth=8,leaf=20      inner log loss 0.27945   AUC 0.7772
    depth=8,leaf=50      inner log loss 0.28165   AUC 0.7745
    depth=12,leaf=20     inner log loss 0.27961   AUC 0.7767
    depth=12,leaf=50     inner log loss 0.28136   AUC 0.7750
    depth=None,leaf=20   inner log loss 0.27953   AUC 0.7768
    depth=None,leaf=50   inner log loss 0.28129   AUC 0.7753

  tuning XGBoost (6 configurations)
    depth=3,mcw=5        inner log loss 0.27661   AUC 0.7829
    depth=3,mcw=20       inner log loss 0.27848   AUC 0.7795
    depth=4,mcw=5        inner log loss 0.27676   AUC 0.7818
    depth=4,mcw=20       inner log loss 0.27817   AUC 0.7799
    depth=5,mcw=5        inner log loss 0.27784   AUC 0.7795
    depth=5,mcw=20       inner log loss 0.27874   AUC 0.7782

  tuning CatBoost (4 configurations)
    depth=4,l2=3         inner log loss 0.27539   AUC 0.7862
    depth=4,l2=10        inner log loss 0.27457   AUC 0.7880
    depth=6,l2=3         inner log loss 0.27556   AUC 0.7850
    depth=6,l2=10        inner log loss 0.27594   AUC 0.7841

  selected main model: CatBoost (depth=4,l2=10) on inner-validation log loss 0.27457
  main model AUC 0.7747 [0.7546, 0.7930]
  main model Brier 0.07766 [0.07343, 0.08211]
  [table] outputs/tables/table3_shot_quality_models.csv  (7 rows)
  [table] outputs/tables/table3b_shot_quality_by_subset.csv  (3 rows)

  generating out-of-fold shot quality (5-fold, grouped by match)
    fold 1/5 done
    fold 2/5 done
    fold 3/5 done
    fold 4/5 done
    fold 5/5 done
  out-of-fold: AUC 0.7800, calibration slope 1.012, intercept -0.039, mean SQ 0.1098 vs observed goal rate 0.1067
  [data ] outputs/data/shots_scored.parquet  (21,311 rows x 98 cols)
  [fig  ] outputs/figures/figure2_calibration.pdf / .png
  [fig  ] outputs/figures/figure3_shap_importance.pdf / .png
```

### `logs/s01b.log`

```text

==============================================================================
STEP 1b  Cross-season continuity of the event-data taxonomy
==============================================================================
              Variable                 Type  2022/23  2023/24  2024/25  Ratio 2024/25 : earlier seasons
  situation = assisted Technical annotation   0.5574   0.5452   0.4678                             0.85
    situation = corner Technical annotation   0.1550   0.1592   0.1650                             1.05
 situation = open_play Technical annotation   0.1586   0.1490   0.1349                             0.88
 situation = set_piece Technical annotation   0.0635   0.0694   0.0684                             1.03
  situation = throw_in Technical annotation   0.0112   0.0139   0.0651                             5.18
 situation = free_kick Technical annotation   0.0278   0.0227   0.0189                             0.75
situation = fast_break Technical annotation   0.0158   0.0295   0.0703                             3.10
   situation = penalty Technical annotation   0.0107   0.0110   0.0097                             0.90
       mean_distance_m    Spatial / derived  18.1188  18.0029  17.0413                             0.94
     close_range_share    Spatial / derived   0.1982   0.2125   0.2620                             1.28
      inside_box_share    Spatial / derived   0.6186   0.6063   0.6322                             1.03
          header_share    Spatial / derived   0.1966   0.1840   0.1867                             0.98
  [table] outputs/tables/table2b_taxonomy_continuity.csv  (12 rows)

  2 variables shift by more than 50% at the season boundary:
              Variable  Ratio 2024/25 : earlier seasons
  situation = throw_in                             5.18
situation = fast_break                             3.10

  step test (mean before vs after the 2024/25 season boundary):
    fast_break   before 0.0223 (SD 0.0103), after 0.0693 (SD 0.0114), Welch p = 5.15e-09, non-overlapping ranges: yes
    throw_in     before 0.0123 (SD 0.0054), after 0.0640 (SD 0.0119), Welch p = 5.33e-08, non-overlapping ranges: yes
    assisted     before 0.5518 (SD 0.0227), after 0.4728 (SD 0.0385), Welch p = 5.86e-05, non-overlapping ranges: no
    distance     before 18.0591 (SD 0.4085), after 17.0841 (SD 0.4433), Welch p = 2.10e-05, non-overlapping ranges: no
  [fig  ] outputs/figures/figureS3_taxonomy_break.pdf / .png
```

### `logs/s04_profiles.log`

```text

==============================================================================
STEP 4  Team and player attacking-profile construction
==============================================================================
  team-match profiles: 1,652
  team-season profiles: 48 (>= 20 matches; {'2022/23': 16, '2023/24': 16, '2024/25': 16})
  player-season profiles: 303 (>= 25 shots)
  no missing values in the primary profile feature block

  short-rest coverage: 2 of 48 team-seasons had no short-rest match (median 2 such matches)
  [data ] outputs/data/team_match_profiles.parquet  (1,652 rows x 55 cols)
  [data ] outputs/data/team_season_profiles.parquet  (48 rows x 42 cols)
  [data ] outputs/data/player_season_profiles.parquet  (303 rows x 28 cols)
  [table] outputs/tables/tableS1_player_profiles_top.csv  (20 rows)

  team-season profile summary (primary clustering features):
                            mean    std     min     max
mean_shot_quality          0.109  0.014   0.085   0.150
median_shot_quality        0.065  0.011   0.045   0.096
xg_per_match               1.415  0.388   0.875   2.715
goals_minus_xg_per_match  -0.040  0.252  -0.572   0.862
mean_distance_m           17.791  0.961  15.613  19.929
mean_angle_proxy           0.589  0.028   0.530   0.643
high_quality_share         0.188  0.045   0.110   0.306
low_quality_share          0.395  0.063   0.246   0.541
shots_per_match           12.890  2.150   9.571  18.457
on_target_rate             0.337  0.032   0.269   0.406
blocked_rate               0.260  0.029   0.192   0.315
goal_rate                  0.105  0.022   0.043   0.175
header_share               0.188  0.034   0.112   0.264
assisted_share             0.522  0.059   0.402   0.647
set_piece_share            0.291  0.047   0.170   0.413
close_range_share          0.222  0.044   0.138   0.334
fast_break_share           0.038  0.026   0.003   0.106
```

### `logs/s05.log`

```text

==============================================================================
STEP 5  Attacking-profile structure and clustering
==============================================================================
  15 feature pairs in the full block correlate above |r| = 0.80
  [table] outputs/tables/table4e_feature_redundancy.csv  (15 rows)
  Full (17 features):
    PC1 42.9%   PC2 16.5%   PC1+PC2 59.4%   PC3 11.1%   PC4 8.3%
    eigenvalues: PC1 7.44  PC2 2.87  PC3 1.92  PC4 1.44  PC5 0.92
    scree gaps (pp): PC1->PC2 26.4, PC2->PC3 5.4, PC3->PC4 2.8
    components with eigenvalue > 1 (Kaiser): 4; retained for 80% of variance: 5
    NOTE: the first two components do not dominate - PC3 exceeds the Kaiser threshold, so the two-component plot is a projection for readability, not a claim that the profile space is two-dimensional.
  Reduced (9 features):
    PC1 30.2%   PC2 18.6%   PC1+PC2 48.9%   PC3 14.6%   PC4 12.8%
    eigenvalues: PC1 2.78  PC2 1.71  PC3 1.34  PC4 1.18  PC5 0.71
    scree gaps (pp): PC1->PC2 11.6, PC2->PC3 4.1, PC3->PC4 1.8
    components with eigenvalue > 1 (Kaiser): 4; retained for 80% of variance: 5
    NOTE: the first two components do not dominate - PC3 exceeds the Kaiser threshold, so the two-component plot is a projection for readability, not a claim that the profile space is two-dimensional.
  [table] outputs/tables/table4g_pca_scree.csv  (26 rows)

  cluster-selection metrics:
         Feature set Algorithm  k  Silhouette  Calinski-Harabasz  Davies-Bouldin  BIC (GMM)  Mean bootstrap Jaccard  Min bootstrap Jaccard  Smallest cluster n
  Full (17 features)   k-means  2       0.418             27.084           0.888        NaN                   0.602                  0.600                   8
  Full (17 features)   k-means  3       0.216             21.381           1.513        NaN                   0.510                  0.450                   8
  Full (17 features)   k-means  4       0.197             18.797           1.389        NaN                   0.446                  0.357                   8
  Full (17 features)   k-means  5       0.205             16.749           1.361        NaN                   0.415                  0.343                   8
  Full (17 features)   k-means  6       0.201             15.661           1.207        NaN                   0.418                  0.316                   2
  Full (17 features)   k-means  7       0.205             15.051           1.222        NaN                   0.429                  0.281                   2
  Full (17 features)       GMM  2       0.418             27.084           0.888    916.783                   0.536                  0.501                   8
  Full (17 features)       GMM  3       0.134             16.182           1.803    939.612                   0.398                  0.202                   7
  Full (17 features)       GMM  4       0.075             12.590           2.195    953.402                   0.352                  0.222                   6
  Full (17 features)       GMM  5       0.124             12.062           1.486    971.813                   0.356                  0.260                   2
  Full (17 features)       GMM  6       0.091             10.376           1.274    963.992                   0.344                  0.216                   1
  Full (17 features)       GMM  7       0.133             10.981           1.365    991.098                   0.321                  0.251                   1
  Full (17 features)      Ward  2       0.418             27.084           0.888        NaN                   0.626                  0.620                   8
  Full (17 features)      Ward  3       0.175             19.049           1.760        NaN                   0.467                  0.378                   8
  Full (17 features)      Ward  4       0.179             16.488           1.551        NaN                   0.416                  0.327                   7
  Full (17 features)      Ward  5       0.195             15.461           1.377        NaN                   0.397                  0.305                   7
  Full (17 features)      Ward  6       0.199             14.742           1.247        NaN                   0.430                  0.316                   2
  Full (17 features)      Ward  7       0.218             14.555           1.154        NaN                   0.460                  0.313                   2
Reduced (9 features)   k-means  2       0.198             14.727           1.678        NaN                   0.503                  0.493                  22
Reduced (9 features)   k-means  3       0.193             12.887           1.595        NaN                   0.428                  0.417                   7
Reduced (9 features)   k-means  4       0.191             12.166           1.384        NaN                   0.404                  0.361                   7
Reduced (9 features)   k-means  5       0.197             11.733           1.369        NaN                   0.396                  0.300                   7
Reduced (9 features)   k-means  6       0.232             11.855           1.172        NaN                   0.447                  0.393                   4
Reduced (9 features)   k-means  7       0.254             12.254           1.113        NaN                   0.477                  0.427                   2
Reduced (9 features)       GMM  2       0.183              9.339           1.871    821.115                   0.379                  0.271                  12
Reduced (9 features)       GMM  3       0.099              8.064           2.097    840.475                   0.313                  0.255                   9
Reduced (9 features)       GMM  4       0.035              5.243           2.236    853.176                   0.285                  0.222                   5
Reduced (9 features)       GMM  5       0.137              7.517           1.217    868.474                   0.300                  0.190                   1
Reduced (9 features)       GMM  6       0.116              6.978           1.423    881.451                   0.323                  0.268                   1
Reduced (9 features)       GMM  7       0.216             10.279           1.112    885.292                   0.389                  0.323                   1
Reduced (9 features)      Ward  2       0.193             13.789           1.715        NaN                   0.438                  0.409                  19
Reduced (9 features)      Ward  3       0.163             11.340           1.594        NaN                   0.400                  0.388                   7
Reduced (9 features)      Ward  4       0.167             11.264           1.537        NaN                   0.400                  0.335                   7
Reduced (9 features)      Ward  5       0.198             11.242           1.405        NaN                   0.422                  0.345                   4
Reduced (9 features)      Ward  6       0.200             11.112           1.272        NaN                   0.426                  0.366                   4
Reduced (9 features)      Ward  7       0.226             11.464           1.156        NaN                   0.471                  0.384                   2
  [table] outputs/tables/table4b_cluster_selection_metrics.csv  (36 rows)

  gap statistic (can select k = 1, i.e. no cluster structure):
    Full (17 features): k selected = 1
             Feature set  k  log_Wk   gap   s_k
      Full (17 features)  1   6.531 0.463 0.065
      Full (17 features)  2   6.068 0.478 0.060
      Full (17 features)  3   5.863 0.487 0.061
      Full (17 features)  4   5.706 0.479 0.062
      Full (17 features)  5   5.591 0.468 0.074
      Full (17 features)  6   5.478 0.458 0.070
      Full (17 features)  7   5.367 0.451 0.077
    Reduced (9 features): k selected = 1
/Users/simonsalaj/.venvs/football/lib/python3.12/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
               Feature set  k  log_Wk   gap   s_k
      Reduced (9 features)  1   5.893 0.532 0.060
      Reduced (9 features)  2   5.615 0.443 0.060
      Reduced (9 features)  3   5.440 0.418 0.062
      Reduced (9 features)  4   5.289 0.404 0.062
      Reduced (9 features)  5   5.155 0.413 0.076
      Reduced (9 features)  6   5.013 0.436 0.072
      Reduced (9 features)  7   4.866 0.465 0.077
  [table] outputs/tables/table4f_gap_statistic.csv  (14 rows)

  best available partition: k-means, k=2 on the full (17 features) block (silhouette 0.418, min bootstrap Jaccard 0.600)
  null reference (no clusters by construction): silhouette 0.300, mean bootstrap Jaccard 0.560
  HYPOTHESIS 1 VERDICT: NOT supported - profiles form a continuum; the partition below is reported as a descriptive device, not as discrete types
  agreement with alternative algorithms (ARI): {'GMM': 1.0, 'Ward': 1.0}

  cluster labels:
    0: High-volume chance creators            n= 8  bootstrap Jaccard 0.60
    1: Main body of the league                n=40  bootstrap Jaccard 0.60
  [table] outputs/tables/table4d_pca_loadings.csv  (17 rows)

  chance-creation axis (PC1, 42.9%), strongest loadings:
                     PC1 chance creation (42.9%)
mean_shot_quality                          0.348
high_quality_share                         0.340
median_shot_quality                        0.336
xg_per_match                               0.333
low_quality_share                         -0.330

  finishing axis (PC2, 16.5%), strongest loadings:
                          PC2 finishing (16.5%)
header_share                             -0.450
on_target_rate                            0.358
goals_minus_xg_per_match                  0.352
assisted_share                            0.341
close_range_share                        -0.325

  chance-source axis (PC3, 11.1%), strongest loadings:
                          PC3 chance source (11.1%)
set_piece_share                               0.552
assisted_share                               -0.520
goals_minus_xg_per_match                      0.341
mean_angle_proxy                              0.300
shots_per_match                              -0.231
  [table] outputs/tables/table4h_component_invariance.csv  (3 rows)

  partition stability across the number of retained components:
     Components A  Components B  Adjusted Rand index
                2             3                  1.0
                2             4                  1.0
                3             4                  1.0
  [table] outputs/tables/table4_cluster_characteristics.csv  (2 rows)
  [table] outputs/tables/table4c_cluster_membership.csv  (48 rows)
  [data ] outputs/data/team_season_clustered.parquet  (48 rows x 48 cols)
  [fig  ] outputs/figures/figure4_profile_space.pdf / .png
  [fig  ] outputs/figures/figure5_cluster_radar.pdf / .png
```

### `logs/s06.log`

```text

==============================================================================
STEP 6  Temporal stability and transition analysis
==============================================================================
  18 teams, 14 present in all three seasons

  consecutive-season transitions: n = 30
  same-cluster retention rate: 0.967
  Cramer's V (from-cluster x to-cluster): 0.890
  mean adjusted Rand index between consecutive seasons: 0.842
  retention expected by chance: 0.700
  [table] outputs/tables/table5_transition_matrix.csv  (3 rows)

  season-to-season repeatability (profile axes first):
                 Feature  n team-season pairs  Year-to-year r        95% CI      p  ICC (team)
    chance_creation_axis                   30           0.764  [0.56, 0.88] 0.0000       0.780
          finishing_axis                   30           0.232 [-0.14, 0.55] 0.2177       0.205
      chance_source_axis                   30           0.383  [0.03, 0.65] 0.0367       0.331
       mean_shot_quality                   30           0.673  [0.41, 0.83] 0.0000       0.273
     median_shot_quality                   30           0.710  [0.47, 0.85] 0.0000       0.234
            xg_per_match                   30           0.885  [0.77, 0.94] 0.0000       0.726
goals_minus_xg_per_match                   30           0.380  [0.02, 0.65] 0.0385      -0.243
         mean_distance_m                   30           0.658  [0.39, 0.82] 0.0001       0.358
  [table] outputs/tables/table5b_profile_repeatability.csv  (20 rows)

  own-team profile distance 4.04 (SD 1.07) vs different-team null 5.68 (SD 1.83); permutation p = 0.0001

  trajectory typology for 14 teams:
Trajectory
Oscillating profile                         8
Oscillating profile (load-sensitive)        2
Stable profile (load-sensitive)             1
Progressive improvement (load-sensitive)    1
Progressive improvement                     1
Stable profile                              1
  [table] outputs/tables/table5c_trajectory_typology.csv  (14 rows)
  [fig  ] outputs/figures/figure6_profile_transitions.pdf / .png
```

### `logs/s07.log`

```text

==============================================================================
STEP 7  Competitive-load analysis
==============================================================================
  20,782 shots with complete calendar-load information
  exposure distribution: short rest 6.1% of shots, mean matches in previous 14 days 1.82
    fitted all outcomes for exposure: Short rest (<= 3 days vs >= 4)
    fitted all outcomes for exposure: Matches in previous 14 days (per match)
    fitted all outcomes for exposure: Days since previous fixture (per day, capped at 14)

  primary outcome (shot quality):
                                           Exposure              Outcome  Estimate  CI low  CI high      p
                     Short rest (<= 3 days vs >= 4) Shot quality (logit)   -0.0195 -0.0761   0.0372 0.5003
            Matches in previous 14 days (per match) Shot quality (logit)    0.0005 -0.0177   0.0187 0.9560
Days since previous fixture (per day, capped at 14) Shot quality (logit)    0.0052 -0.0008   0.0111 0.0907

  secondary outcomes with q < 0.05:
                                           Exposure           Outcome  Estimate  CI low  CI high  q (BH-FDR)
Days since previous fixture (per day, capped at 14) Shot distance (m)    -0.077 -0.1251   -0.029      0.0352
  [table] outputs/tables/table6_load_effects.csv  (24 rows)

  short-rest effect on shot quality under three calendar definitions:
                    Calendar definition  Short-rest shots  Estimate  CI low  CI high      p
                   League-only calendar               693    0.0026 -0.0725   0.0777 0.9463
               All-competition calendar              1260   -0.0195 -0.0761   0.0372 0.5003
All-competition, alternative cut-points              4326    0.0039 -0.0307   0.0385 0.8266
  [table] outputs/tables/table6b_calendar_definitions.csv  (3 rows)

  broader competitive-load exposures (shot quality):
                                   Exposure  Exposed shots  Estimate  CI low  CI high      p
          European match in previous 7 days           2022    0.0099 -0.0445   0.0643 0.7206
               Cup match in previous 7 days           1624   -0.0582 -0.1086  -0.0079 0.0233
Upcoming cup/European fixture within 3 days            614   -0.0237 -0.1032   0.0559 0.5597
         European participant (team-season)           6760    0.0799  0.0307   0.1291 0.0015
  [table] outputs/tables/table6c_european_load.csv  (4 rows)
  [fig  ] outputs/figures/figure7_shot_quality_by_rest.pdf / .png
  [fig  ] outputs/figures/figure8_late_game_trends.pdf / .png
  [fig  ] outputs/figures/figure9_team_resilience.pdf / .png
  [table] outputs/tables/table6d_team_resilience.csv  (14 rows)
```

### `logs/s08_prediction.log`

```text

==============================================================================
STEP 8  Early-season prediction
==============================================================================
  [table] outputs/tables/tableS2_stabilisation_curves.csv  (56 rows)
  matches needed to reach Pearson r >= 0.70 with remainder-season performance:
Feature
xg_per_match             5.0
shots_per_match         10.0
chance_creation_axis    12.0

  First 5 matches: 48 early profiles, 48 remainder profiles

  First 10 matches: 48 early profiles, 48 remainder profiles

  First 30% of season: 48 early profiles, 48 remainder profiles
  [table] outputs/tables/tableS3_early_season_regression.csv  (36 rows)

  early-to-remainder prediction:
             Window                   Target                      Model  Holdout R2 (2024/25 remainder)  Holdout MAE  Holdout Spearman rho
    First 5 matches     chance creation axis Naive early-value baseline                          -1.387        3.619                 0.582
    First 5 matches     chance creation axis           Ridge regression                          -0.761        3.013                 0.597
    First 5 matches           finishing axis Naive early-value baseline                          -1.156        1.790                 0.444
    First 5 matches           finishing axis           Ridge regression                          -2.315        2.463                 0.582
    First 5 matches             xg per match Naive early-value baseline                          -0.015        0.335                 0.788
    First 5 matches             xg per match           Ridge regression                           0.316        0.216                 0.847
    First 5 matches goals minus xg per match Naive early-value baseline                          -1.041        0.245                 0.326
    First 5 matches goals minus xg per match           Ridge regression                          -3.505        0.385                 0.303
   First 10 matches     chance creation axis Naive early-value baseline                          -0.103        2.417                 0.612
   First 10 matches     chance creation axis           Ridge regression                          -0.540        3.021                 0.553
   First 10 matches           finishing axis Naive early-value baseline                          -1.152        1.877                 0.594
   First 10 matches           finishing axis           Ridge regression                          -1.040        1.873                 0.676
   First 10 matches             xg per match Naive early-value baseline                           0.377        0.303                 0.726
   First 10 matches             xg per match           Ridge regression                           0.321        0.282                 0.600
   First 10 matches goals minus xg per match Naive early-value baseline                          -0.961        0.273                 0.003
   First 10 matches goals minus xg per match           Ridge regression                          -2.209        0.356                 0.356
First 30% of season     chance creation axis Naive early-value baseline                          -0.205        2.542                 0.521
First 30% of season     chance creation axis           Ridge regression                          -0.537        3.085                 0.529
First 30% of season           finishing axis Naive early-value baseline                          -0.635        1.465                 0.750
First 30% of season           finishing axis           Ridge regression                          -0.762        1.682                 0.735
First 30% of season             xg per match Naive early-value baseline                           0.429        0.299                 0.726
First 30% of season             xg per match           Ridge regression                           0.366        0.276                 0.629
First 30% of season goals minus xg per match Naive early-value baseline                          -0.874        0.250                 0.203
First 30% of season goals minus xg per match           Ridge regression                          -2.297        0.347                 0.329
  [fig  ] outputs/figures/figureS1_stabilisation.pdf / .png
  [fig  ] outputs/figures/figureS2_early_season_shap.pdf / .png
```

### `logs/s09_robustness.log`

```text

==============================================================================
A  Shot-quality model sensitivity
==============================================================================
  reference AUC 0.7747, BSS 0.1461
  Excluding penalties              AUC 0.7591  BSS 0.1065
  Excluding added-time shots       AUC 0.7729  BSS 0.1425
  Excluding set-piece shots        AUC 0.7561  BSS 0.1116
  Open play only                   AUC 0.7561  BSS 0.1116
  Metric-corrected geometry        AUC 0.7751  BSS 0.1445
  Excluding situation predictor    AUC 0.7751  calib. intercept -0.275 (ref -0.320)
  calibration 2022/23: slope 1.038, intercept +0.155, obs/pred 1.123
  calibration 2023/24: slope 1.060, intercept -0.027, obs/pred 0.980
  calibration 2024/25: slope 0.982, intercept -0.237, obs/pred 0.839
  train 2022/23 -> test 2023/24: AUC 0.7885
  train 2023/24 -> test 2024/25: AUC 0.7776

==============================================================================
B  Clustering sensitivity
==============================================================================
  All profile features (primary)                 silhouette 0.418  min Jaccard 0.600
  Spatial features only                          silhouette 0.392  min Jaccard 0.538
  Shot-quality features only                     silhouette 0.565  min Jaccard 0.621
  Shot-selection features only                   silhouette 0.206  min Jaccard 0.436
  Profile + competitive-load response features   silhouette 0.429  min Jaccard 0.606
  Within-season standardisation (primary)        min Jaccard 0.600
  Pooled standardisation                         min Jaccard 0.543

==============================================================================
B2  No-situation profile-construction workflow
==============================================================================
    no-situation profile workflow fold 1/5 done
    no-situation profile workflow fold 2/5 done
    no-situation profile workflow fold 3/5 done
    no-situation profile workflow fold 4/5 done
    no-situation profile workflow fold 5/5 done
  [table] outputs/tables/table7b_no_situation_profile_comparison.csv  (14 rows)
  no-situation workflow: 48 team-seasons, ARI 1.000, silhouette 0.485, min Jaccard 0.591

==============================================================================
C  Competitive-load model sensitivity
==============================================================================
  reference short-rest effect -0.0195 (SE 0.0289, p = 0.500)
  Alternative rest cut-points (<=4 / 5-7 / >=8 days) +0.0039 (p = 0.827, n = 20,782)
  League-only calendar                               +0.0022 (p = 0.954, n = 20,782)
  Excluding post-break fixtures                      -0.0195 (p = 0.501, n = 19,585)
  Excluding penalties                                -0.0212 (p = 0.430, n = 20,564)
  Open play only                                     -0.0172 (p = 0.599, n = 14,750)
  European participants only                         -0.0816 (p = 0.107, n = 6,760)
  Non-European participants only                     +0.0115 (p = 0.745, n = 14,022)
  Adjusting for cup + European exposure              +0.0085 (p = 0.787)
  Team + opponent fixed effects                      -0.0262 (p = 0.392)
  No team identifiers                                -0.0177 (p = 0.565)
  season 2022/23                                     -0.1786 (p = 0.005)
  season 2023/24                                     +0.0335 (p = 0.479)
  season 2024/25                                     +0.0057 (p = 0.901)

==============================================================================
D  Player-profile threshold sensitivity
==============================================================================
  >= 25 shots: 303 player-seasons, SD(finishing over expectation) 2.144, r(shot quality, goal rate) 0.635
  >= 30 shots: 234 player-seasons, SD(finishing over expectation) 2.314, r(shot quality, goal rate) 0.632
  >= 40 shots: 124 player-seasons, SD(finishing over expectation) 2.589, r(shot quality, goal rate) 0.712
  >= 50 shots:  70 player-seasons, SD(finishing over expectation) 2.877, r(shot quality, goal rate) 0.726
  [table] outputs/tables/table7_robustness.csv  (68 rows)

  68 robustness rows written
```

### `logs/s10_figures_summary.log`

```text

==============================================================================
STEP 10  Figure 1 and results summary
==============================================================================
  21,255 of 21,311 shots fall inside the plotted attacking third
  [fig  ] outputs/figures/figure1_shot_density.pdf / .png
  [doc  ] outputs/RESULTS_SUMMARY.md (83 lines)
```

### `logs/s12_core_validation.log`

```text

==============================================================================
STEP 12  Core validation analyses
==============================================================================
  [table] outputs/tables/table8_split_half_reliability.csv  (9 rows)
  [table] outputs/tables/table8b_split_half_differences.csv  (2 rows)

  split-half reliability:
                 Feature  Pearson r Pearson 95% bootstrap CI  Spearman rho Spearman 95% bootstrap CI  MAE first vs second half  n team-seasons
         shots_per_match      0.637           [0.430, 0.775]         0.541            [0.254, 0.731]                     1.651              48
            xg_per_match      0.751           [0.572, 0.868]         0.676            [0.453, 0.812]                     0.243              48
       mean_shot_quality      0.593           [0.355, 0.757]         0.571            [0.310, 0.745]                     0.013              48
      high_quality_share      0.602           [0.406, 0.765]         0.561            [0.331, 0.741]                     0.040              48
    chance_creation_axis      0.628           [0.418, 0.773]         0.506            [0.232, 0.716]                     2.209              48
         goals_per_match      0.626           [0.412, 0.779]         0.526            [0.247, 0.737]                     0.340              48
goals_minus_xg_per_match      0.452           [0.124, 0.665]         0.322           [-0.005, 0.600]                     0.278              48
          on_target_rate      0.381           [0.143, 0.594]         0.375            [0.108, 0.616]                     0.035              48
          finishing_axis      0.552           [0.341, 0.711]         0.557            [0.310, 0.730]                     1.641              48
                                 Comparison  Pearson r difference 95% paired cluster bootstrap CI Two-sided paired bootstrap p
  chance_creation_axis minus finishing_axis                 0.076                 [-0.354, 0.282]                        0.663
xg_per_match minus goals_minus_xg_per_match                 0.299                 [-0.039, 0.618]                        0.067
  [table] outputs/tables/table9_practical_validity.csv  (10 rows)

  practical validity models:
                         Outcome                                   Term  Estimate  SE cluster(team) p cluster(team) Team-cluster bootstrap 95% CI Team-cluster bootstrap p  Model R2                                                                              Inference note
                points_per_match           process_chance_creation_axis    0.1732            0.0157          < .001                [0.132, 0.201]                   < .001     0.815 Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
                points_per_match                         finishing_axis    0.1076            0.0213          < .001                [0.060, 0.149]                   < .001     0.815 Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
       goal_difference_per_match           process_chance_creation_axis    0.2745            0.0253          < .001                [0.212, 0.317]                   < .001     0.845 Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
       goal_difference_per_match                         finishing_axis    0.1814            0.0350          < .001                [0.100, 0.245]                   < .001     0.845 Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
nonpen_goal_difference_per_match           process_chance_creation_axis    0.2501            0.0197          < .001                [0.198, 0.284]                   < .001     0.836 Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
nonpen_goal_difference_per_match                         finishing_axis    0.1582            0.0328          < .001                [0.083, 0.216]                   < .001     0.836 Exploratory concurrent association; team-cluster bootstrap checks small-cluster sensitivity
                points_per_match Spearman(process_chance_creation_axis)    0.6662               NaN                                                                              NaN                                                                Descriptive rank association
       goal_difference_per_match Spearman(process_chance_creation_axis)    0.7120               NaN                                                                              NaN                                                                Descriptive rank association
                points_per_match               Spearman(finishing_axis)    0.1024               NaN                                                                              NaN                                                                Descriptive rank association
       goal_difference_per_match               Spearman(finishing_axis)    0.0973               NaN                                                                              NaN                                                                Descriptive rank association
  [table] outputs/tables/table9b_opponent_adjusted_profiles.csv  (48 rows)

  opponent-adjusted xG rating vs chance-creation axis:
 Spearman adjusted xG vs chance axis
                               0.721
  [table] outputs/tables/table3c_temporal_validation.csv  (2 rows)

  temporal shot-quality validation:
                                               Model  N shots    AUC  PR-AUC  Log loss  Brier score  Brier skill score  Calibration slope  Calibration intercept    ECE
 Temporal development validation: 2022/23 -> 2023/24     7261 0.7880  0.4133    0.2746       0.0773             0.1797             1.0602                -0.1884 0.0158
Final temporal holdout: 2022/23 + 2023/24 -> 2024/25     7102 0.7747  0.3830    0.2764       0.0777             0.1461             0.9805                -0.3199 0.0269
  [fig  ] outputs/figures/figure10_split_half_reliability.pdf / .png
  [fig  ] outputs/figures/figure11_practical_validity.pdf / .png
```
