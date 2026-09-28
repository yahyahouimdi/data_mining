# MAT → CSV Conversion Report

Script: `converter.py` — folder: `data_mining/project/data`

## 1. Summary

| Item | Value |
|---|---|
| MAT files found | **44** |
| Recordings (pairs) | **22** (one `*d.mat` + one `*h.mat` each) |
| Signal files (`*d`, variable `output_d`) | 22 → 22 CSV |
| Annotation files (`*h`, variable `output_h`) | 22 → 22 CSV |
| Total signal rows written | **8,506,368** |
| Total annotations | **1,057** |
| Sampling rate (inferred from `Time` step 0.003906 s) | **256 Hz** |
| Total recorded duration | ≈ **9.2 h** |
| Status | Conversion finished, all 44 files saved |

## 2. How the conversion transforms the data

**Signal files (`*d.mat`)**

- Original shape: `(N seconds, C channels)`.
- Each original row is expanded into **256 samples** (256 Hz), so `Final rows = N × 256`.
- A `Time` column is added (`0.000000, 0.003906, 0.007812, …`), so `Final columns = C + 1`.
- Example: `190304A_E_0000d` — `(1575, 23)` → `(403200, 24)`  (1575 × 256 = 403,200).

**Annotation files (`*h.mat`)**

- Original shape: `(M, 2)` → final `(M, 3)` with columns `Time`, `Annotations`, `Duration`.

## 3. Signal files (`*d`)

| Recording | Original shape | Final shape | Channels | Duration (min) |
|---|---|---|---|---|
| 190304A_E | (1575, 23) | (403200, 24) | 23 | 26.3 |
| 191113A_D | (1810, 23) | (463360, 24) | 23 | 30.2 |
| 200625A_F | (1788, 12) | (457728, 13) | 12 | 29.8 |
| 210204B_C | (1227, 21) | (314112, 22) | 21 | 20.5 |
| 210208B_G | (1200, 21) | (307200, 22) | 21 | 20.0 |
| 210406A_F | (1454, 23) | (372224, 24) | 23 | 24.2 |
| 210427B_C | (1246, 21) | (318976, 22) | 21 | 20.8 |
| 210504B_C | (3121, 21) | (798976, 22) | 21 | 52.0 |
| 210914B_A | (1205, 21) | (308480, 22) | 21 | 20.1 |
| 210928B_B | (1344, 21) | (344064, 22) | 21 | 22.4 |
| 211027B_C | (1255, 21) | (321280, 22) | 21 | 20.9 |
| 211104B_D | (1210, 21) | (309760, 22) | 21 | 20.2 |
| 220106A_A | (1221, 23) | (312576, 24) | 23 | 20.4 |
| 220121B_D | (1292, 21) | (330752, 22) | 21 | 21.5 |
| 230102B_F | (1225, 21) | (313600, 22) | 21 | 20.4 |
| 230105B_H | (3600, 21) | (921600, 22) | 21 | 60.0 |
| 230210B_D | (1173, 21) | (300288, 22) | 21 | 19.6 |
| 230224B_B | (1428, 21) | (365568, 22) | 21 | 23.8 |
| 230406B_A | (1225, 21) | (313600, 22) | 21 | 20.4 |
| 230515B_G | (1207, 21) | (308992, 22) | 21 | 20.1 |
| 230519A_E | (1216, 23) | (311296, 24) | 23 | 20.3 |
| 230718A_A | (1206, 23) | (308736, 24) | 23 | 20.1 |

## 4. Annotation files (`*h`)

| Recording | Original shape | Final shape | # annotations |
|---|---|---|---|
| 190304A_E | (77, 2) | (77, 3) | 77 |
| 191113A_D | (104, 2) | (104, 3) | 104 |
| 200625A_F | (49, 2) | (49, 3) | 49 |
| 210204B_C | (35, 2) | (35, 3) | 35 |
| 210208B_G | (45, 2) | (45, 3) | 45 |
| 210406A_F | (43, 2) | (43, 3) | 43 |
| 210427B_C | (43, 2) | (43, 3) | 43 |
| 210504B_C | (40, 2) | (40, 3) | 40 |
| 210914B_A | (17, 2) | (17, 3) | 17 |
| 210928B_B | (62, 2) | (62, 3) | 62 |
| 211027B_C | (13, 2) | (13, 3) | 13 |
| 211104B_D | (37, 2) | (37, 3) | 37 |
| 220106A_A | (60, 2) | (60, 3) | 60 |
| 220121B_D | (41, 2) | (41, 3) | 41 |
| 230102B_F | (37, 2) | (37, 3) | 37 |
| 230105B_H | (104, 2) | (104, 3) | 104 |
| 230210B_D | (51, 2) | (51, 3) | 51 |
| 230224B_B | (76, 2) | (76, 3) | 76 |
| 230406B_A | (49, 2) | (49, 3) | 49 |
| 230515B_G | (27, 2) | (27, 3) | 27 |
| 230519A_E | (40, 2) | (40, 3) | 40 |
| 230718A_A | (7, 2) | (7, 3) | 7 |

## 5. Channel layouts found

The recordings do **not** share a single column layout:

| Layout | Columns (after `Time`) | Recordings |
|---|---|---|
| **A — 21 ch** | 19 EEG (`EEGFp1 … EEGO1`) + `SLI` + `ECG` | 15 files (e.g. 210204B_C, 230105B_H) |
| **B — 23 ch, "Fp2-first"** | `EEGFp2, EEGF8, EEGT4, EEGT6, EEGF4, …, EEGT3, EEGT5, ECG, EMG_1_, SLI, EMG_2` | 210406A_F, 220106A_A, 230519A_E, 230718A_A |
| **C — 23 ch, EMG before SLI** | `EEGFp1, EEGFp2, EEGF7, EEGF3, EEGFz, …, EEGO1, EEGO2, EMG1, EMG2, SLI, ECG` | 190304A_E, 191113A_D |
| **D — 12 ch** | `EEGFp1, EEGFp2, EEGC3, EEGC4, EEGO1, …, EEGT3, EEGT4, EMG1, EMG2, SLI, ECG` | 200625A_F |

## 6. Sample output

**Signal CSV** (`210204B_C_0000d.csv`, first rows, columns truncated)

| Time | EEGFp1 | EEGFp2 | EEGF7 | EEGF3 | … | EEGO1 | SLI | ECG |
|---|---|---|---|---|---|---|---|---|
| 0.000000 | 1090.665 | 985.420 | 1212.377 | 373.638 | … | 1088.875 | 0.0895 | 570.346 |
| 0.003906 | 1182.307 | 1053.973 | 1288.089 | 463.848 | … | 1169.062 | 0.0895 | 89.405 |
| 0.007812 | 1664.681 | 1527.397 | 1774.401 | 956.424 | … | 1648.035 | 0.0895 | -170.665 |

**Annotation CSV** (`210204B_C_0000h.csv`)

| Time | Annotations | Duration |
|---|---|---|
| 0.000000 | D��but | NaT |
| 0.003906 | SATURATION DES VOIES : ECG | NaT |
| 0.007812 | HPN 3mn d��but | NaT |
| 0.011719 | ARTEFACT | NaT |
| 0.015625 | HPN - 00:00:20 | NaT |

Frequent annotation labels seen: `Début`, `HPN 3mn début`, `HPN - 00:00:20 / 40 / 1:00…`, `ARTEFACT`, `SATURATION DES VOIES : ECG`, `IMPEDANCE MODE`, `YO` / `YF` (eyes open / closed), `SLI=3Hz … 10Hz` (photic stimulation), `BOUGE`, `absence`, `S'ENDORT`.

## 7. Issues to check

1. **Annotation `Time` is wrong.** In every `*h` file the `Time` column is `0, 0.003906, 0.007812, …` — it is just the row index × 1/256 s, not the real onset of each event. The true annotation times are probably in the original `.mat` (or need to be recovered separately) before you can align annotations with the signals.
2. **`Duration` is entirely `NaT`** — no duration information was extracted.
3. **Encoding problem:** `Début` is written as `D��but`. Save the CSVs with `encoding="utf-8-sig"` (or `latin-1`) and read the source strings with the correct encoding.
4. **Inconsistent channel order/names** across layouts A–D (see section 5). Reorder/rename columns to a common schema before merging files.
5. **Very different amplitude scales.** For example `ECG` reaches ±10⁴ in `200625A_F` and `220106A_A`, while EEG values are ~10²–10³ elsewhere; `SLI` is 19.67 in some files and 0.0895 in others. Normalize per recording/channel.
6. **Flat start in `190304A_E`:** the first rows have the same value (0.038499) in every EEG/EMG channel — likely padding or a disconnected/saturated segment.
7. **Size:** ~8.5 M rows in total; `230105B_H` (921,600 rows) and `210504B_C` (798,976 rows) are the largest. Consider Parquet or chunked reading for modelling.