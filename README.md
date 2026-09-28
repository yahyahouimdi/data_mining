# EEG Seizure Explorer

## Project purpose

This project provides the exploration and annotation part of a Data Mining lab
about automatic detection of absence seizures from EEG recordings.

The implemented workflow is:

```text
MAT files / CSV files
				|
				v
Recording discovery and channel detection
				|
				v
Bounded signal windows + normalization + downsampling
				|
				v
Annotation cleaning and seizure interval validation
				|
				v
Sample-level binary target: 0 = outside seizure, 1 = seizure
				|
				v
CSV / Parquet export for future features and machine learning
```

The application is intentionally structured so feature extraction and
classification can be added after the target-generation stage.

## What was implemented

### Application shell

- Streamlit entry point in `app.py`.
- Wide clinical-dashboard layout with custom CSS.
- Deep blue/teal signal palette, red seizure overlays, and amber warnings.
- French user-facing labels with English code comments and docstrings.
- Loading spinners, empty states, warnings, and readable error messages.
- Configurable data-folder path and reload button.
- Recording selector with signal/annotation availability badges.

### Part I: Apercu / Overview

- Dynamic discovery of all recording pairs in the configured folder.
- Metrics for row count, number of variables, sampling frequency, duration,
	and missing values.
- Sampling frequency calculated from the `Time` column rather than assumed
	blindly, while remaining compatible with the 256 Hz source data.
- First 10 or 20 signal rows displayed in a dataframe.
- Per-channel mean, standard deviation, minimum, maximum, and missing counts.
- Signal viewer with:
	- stacked clinical EEG lanes;
	- overlaid channels;
	- selectable channels grouped as EEG, EMG, ECG, SLI, or Other;
	- optional z-score or min-max normalization;
	- seizure interval overlays;
	- Plotly pan, zoom, unified hover, and range slider;
	- configurable downsampling density.
- Multi-recording availability table.

### Part II: Annotations et crises

- Raw annotation table with label filtering.
- UTF-8 BOM and Latin-1 CSV fallback.
- Repair of common broken labels such as `D��but` to `Début`.
- Explicit warning that `h` file times are row-index times and that their
	durations are not reliable.
- Automatic pairing of configurable absence start/end labels.
- Editable seizure table with:
	- crisis name;
	- start time;
	- end time;
	- automatically calculated duration;
	- dynamic rows for adding and deleting intervals.
- Optional Excel upload through `Annotations.xlsx`.
- Duration statistics: count, mean, minimum, maximum, and mean interval
	between consecutive seizures.
- Gantt-style seizure timeline, duration bar chart, and duration histogram.
- Coefficient of variation and a plain-language answer to whether durations
	are similar. The default threshold is `CV < 0.25`.
- CSV export of validated seizure intervals.

### Part III: Variable cible

- Binary target construction for every sample in the selected window.
- Target value `0` outside seizure intervals and `1` inside them.
- Signal plot with a target step curve below it on a shared time axis.
- Class proportions for normal and seizure samples.
- Total seizure time and class-imbalance warning.
- Export of labeled data as CSV or Parquet.

### Part IV: Comparaison

- Channel selection for spectral analysis.
- Welch power spectral density plot.
- Spectrogram plot based on SciPy.
- Guidance to inspect the approximately 3 Hz spike-and-wave pattern commonly
	associated with absence seizures.

### Part V: Aide

- Short explanations of each application tab.
- Explanation of the `d` signal files and `h` annotation files.
- Explanation of known data limitations and the bounded plotting strategy.

## Data handling and performance

The signal files contain hundreds of thousands of rows and can be hundreds of
megabytes. The application therefore uses the following safeguards:

- Signal columns are discovered from the CSV header; channel names are never
	hard-coded.
- `*_0000d.csv` files are treated as signals and `*_0000h.csv` files as raw
	annotations.
- Selected columns are loaded instead of plotting every available channel.
- Time-window reads use Parquet filters when a cache exists.
- CSV fallback reads use chunks and filter rows by the requested time range.
- CSV files can be converted to Parquet in `data/.cache/` on first use.
- Plotly `Scattergl` traces and min/max bucket decimation retain visible EEG
	spikes without sending the full recording to the browser.
- Streamlit `st.cache_data` caches recording discovery, metadata, and signal
	windows.
- Flat or constant channels are handled without division-by-zero failures.
- Missing and non-numeric values are converted safely during processing.

The first load of a new large CSV can take time because the Parquet cache may
need to be created. Later reads are intended to be much faster.

## Project structure

```text
app.py                         Streamlit launcher
requirements.txt               Runtime and test dependencies
README.md                      Project documentation and verification report

eeg_app/
	__init__.py
	app.py                       Dashboard tabs, state, and exports
	src/
		__init__.py
		io_utils.py                Discovery, encoding, caching, window reads
		channels.py                Dynamic channel grouping, layouts, and units
		annotations.py             Cleaning, pairing, editing, and statistics
		target.py                  Binary target construction
		signal_utils.py            Sampling, normalization, decimation, spectra
		plotting.py                Plotly signal, seizure, target, and duration plots
		ui.py                      CSS injection and reusable metric cards

tests/
	test_core.py                 Four focused unit tests

data/
	*_0000d.csv                  Signal recordings
	*_0000h.csv                  Raw annotation recordings
	.cache/                      Generated Parquet cache, ignored by Git

data/converter.py              Existing MAT-to-CSV conversion utility
```

## Installation and run instructions

### Existing virtual environment

From the repository root on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### Create a new virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens at the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

The sidebar defaults to `data/`, but another folder can be entered. The folder
must contain files following the discovered `*_0000d.csv` and `*_0000h.csv`
naming pattern.

## Dependencies

The dependency list in `requirements.txt` includes:

- `streamlit` for the interactive dashboard;
- `plotly` for interactive charts;
- `pandas` and `numpy` for data handling;
- `pyarrow` for Parquet caching and export;
- `scipy` for Welch PSD and spectrograms;
- `openpyxl` for optional Excel annotation upload;
- `pytest` for the unit tests;
- the original project dependencies used by the MAT conversion workflow.

Python 3.10 or newer is recommended.

## Typical lab workflow

1. Start the application and select the data folder.
2. Choose a recording from the sidebar.
3. Select one or more EEG channels or use a frontal, temporal, or occipital
	 preset.
4. Inspect the signal overview and missing-value statistics.
5. Review the raw annotation labels.
6. Upload authoritative Excel intervals when available, or manually create and
	 edit seizure intervals in the data editor.
7. Check the duration statistics and coefficient-of-variation conclusion.
8. Open **Variable cible** and inspect the normal/seizure class balance.
9. Export the labeled recording as CSV or Parquet.
10. Use the exported target as input for future windowing, feature extraction,
		and classifier development.

## Verification performed

The implementation was checked with the following commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -q
.\.venv\Scripts\python.exe -m compileall -q .\eeg_app .\app.py
.\.venv\Scripts\python.exe -c "import streamlit, plotly, scipy, pandas, pyarrow; print('dependencies-imported')"
```

The unit test result was:

```text
4 passed
```

The tests cover:

- sampling-frequency calculation;
- start/end seizure pairing;
- seizure duration statistics;
- binary target construction.

The Streamlit server was also started successfully in headless mode and
verified to serve the application locally.

## Requirement checklist

- [x] Streamlit + Plotly application.
- [x] Configurable data directory.
- [x] Dynamic recording and channel discovery.
- [x] EEG, EMG, ECG, SLI, and other channel grouping.
- [x] Encoding fallback and broken-label cleanup.
- [x] Visible warning for unreliable `h` annotation times.
- [x] Excel upload and manual seizure interval editing.
- [x] Automatic seizure pairing from configurable label terms.
- [x] Sampling rate, duration, missing values, and statistics cards.
- [x] Normalization, stacked/overlay display, and seizure overlays.
- [x] Downsampling, caching, and bounded time-window loading.
- [x] Seizure timeline, duration plots, histogram, and CV conclusion.
- [x] Binary target generation and class-imbalance warning.
- [x] CSV and Parquet exports.
- [x] Welch PSD and spectrogram comparison.
- [x] Help tab and responsive visual styling.
- [x] Four focused unit tests.
