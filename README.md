# EEG Seizure Explorer

## Project purpose

This project provides the exploration and annotation part of a Data Mining lab
about automatic detection of absence seizures from EEG recordings.

All explanatory time examples in this document use milliseconds (ms). For
example, one sample at 256 Hz is approximately `3.90625 ms`. The source CSV
timestamps remain in seconds internally so sampling, interval matching, and
Plotly synchronization stay numerically correct.

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
- Automatic pairing of the `YO` start and `YF` end markers used by the supplied
	annotation files, so seizure intervals are populated on first load.
- Automatically extracted `YO`/`YF` intervals are provisional because the
	`h`-file `Time` column is an annotation-row index, not a trustworthy event
	clock. They remain editable and are shown with the existing warning until
	corrected or replaced with authoritative Excel intervals.
- When `data/Annotations.xlsx` is available, it is authoritative. The selected
	recording is matched by normalized `Unnamed: 0` ID, `DebutTrace` is used as
	the trace clock origin, and every `DebCEi`/`FinCEi` pair is converted from
	Excel fractional-day time to elapsed CSV seconds.
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

### Part IV: Segmentation

- Configurable windows of 1000, 2000, 5000, 10000, 30000 ms, or a custom duration.
- Configurable overlap of 0%, 25%, 50%, or 75%.
- Sampling-rate-derived samples per window and step size.
- Window-level `Seizure_Ratio` and class assignment at the 50% threshold.
- Window table, class timeline, CSV export, and Parquet export.

### Part V: Extraction des caractéristiques

- Raw-sample temporal features per EEG channel: Mean, Std, Variance, Min, Max,
	Amplitude, RMS, and Energy.
- Class distribution metrics, feature preview, class box plot, and RMS/Energy
	scatter plot.
- Channel-qualified feature names prevent collisions when several EEG channels
	are selected.
- CSV and Parquet feature-dataset exports.

### Comparison

- Channel selection for spectral analysis.
- Welch power spectral density plot.
- Spectrogram plot based on SciPy.
- Guidance to inspect the approximately 3 Hz spike-and-wave pattern commonly
	associated with absence seizures.

### Aide

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
		features.py                Window-level temporal feature extraction
		signal_utils.py            Sampling, normalization, decimation, spectra
		plotting.py                Plotly signal, seizure, target, and duration plots
		ui.py                      CSS injection and reusable metric cards

tests/
	test_core.py                 Nine focused unit tests

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
11 passed
```

The tests cover:

- sampling-frequency calculation;
- start/end seizure pairing;
- seizure duration statistics;
- binary target construction.

The timing-specific tests also cover Excel clock conversion, normalized
recording-ID matching, and the `190304A-E` example interval.

The Streamlit server was also started successfully in headless mode and
verified to serve the application locally.

## Timing alignment contract

The CSV `Time` column is the master signal axis. It contains elapsed seconds
from the beginning of the EEG trace. The application measures the sampling
frequency from the median positive time step instead of assuming 256 Hz.

`Annotations.xlsx` stores clock times as Excel fractional days. For each
selected recording, the application performs:

```python
relative_seconds = (excel_event_time - DebutTrace) * 86400
```

The converted intervals are validated against the measured CSV duration before
they are used for red overlays, `Class`, segmentation, or features. The
**Verification de la synchronisation** expander shows the original clock
values, converted elapsed seconds, signal duration, sampling frequency, and
validation status.

For the real `190304A-E` workbook row, the verified conversion is:

```text
Trace start: 10:13:35
Excel seizure: 10:35:06 -> 10:35:13
CSV interval:  1291.0 s -> 1298.0 s
Duration:      7.0 s
```

When the workbook does not contain the selected recording, the app falls back
to editable provisional `h`-file YO/YF intervals and keeps the warning visible.

## Partie III - Variable cible

The validated seizure table is clipped to the recording duration, sorted, and
merged when intervals overlap. The sample-level dataset uses closed intervals:

```text
Class = 1 when Début crise <= Time <= Fin crise
Class = 0 otherwise
```

Class construction uses sorted timestamp search boundaries and a NumPy
difference array rather than a Python loop over every sample. The interface
shows the number and percentage of samples in each class, a donut chart, the
imbalance ratio, total seizure time, recording duration, and seizure-time
percentage. A minority class below 20% produces a visible warning; the
threshold is defined in the Part III implementation and can be changed easily.

The signal and Class plots share their time axis. Valid seizure periods are
drawn as translucent red background regions from their start to their end in
both the signal panel and the target panel.

## Partie IV - Segmentation

Segmentation transforms the raw sample-level dataset into overlapping temporal
windows. The default configuration is:

```text
Window duration: 2000 ms
Overlap: 50%
At 256 Hz: 512 samples per window and 256 samples per step
```

The application calculates these values from the measured sampling frequency:

```python
samples_per_window = round(window_duration * sampling_frequency)
step_samples = round(samples_per_window * (1 - overlap))
```

Available window durations are 1000, 2000, 5000, 10000, and 30000 ms, with a
custom duration option. Available overlaps are 0%, 25%, 50%, and 75%.

Each window contains:

```text
Window, Start_Time, End_Time, Duration, Seizure_Ratio, Class
```

The window class rule is:

```text
Seizure_Ratio = mean(sample-level Class)
Window Class = 1 when Seizure_Ratio >= 0.50
Window Class = 0 otherwise
```

The window table, class counts, percentages, class timeline, CSV, and Parquet
exports are available in **Partie IV - Segmentation**.

## Partie V - Extraction des caractéristiques

Part V calculates temporal features from the original raw samples in each
window. It never uses the visually downsampled signal. For every selected EEG
channel, the following eight features are calculated:

- **Mean**: arithmetic mean;
- **Std**: standard deviation;
- **Variance**: variance;
- **Min**: minimum value;
- **Max**: maximum value;
- **Amplitude**: `Max - Min`;
- **RMS**: `sqrt(mean(window ** 2))`;
- **Energy**: `sum(window ** 2)`.

NaN and infinite values are removed safely before calculation. Empty windows
return zero-valued features, and constant signals remain valid with zero
standard deviation and variance.

Feature names are qualified by channel so multiple EEG channels never
overwrite one another, for example:

```text
EEGFp1_Mean, EEGFp1_Std, EEGFp1_Variance, EEGFp1_Amplitude,
EEGFp1_RMS, EEGFp1_Energy, EEGFp2_Mean, EEGFp2_Std, ... , Class
```

The feature tab reports total windows, class 0/class 1 counts and percentages,
shows the first 100 feature rows, provides class distribution box plots and an
RMS-versus-Energy scatter plot, and exports the feature dataset as CSV or
Parquet. This dataset is the starting point for the future classifier stage.

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
- [x] Complete Parts III, IV, and V pipeline.
- [x] Red seizure overlays clipped to the visible signal time range.
- [x] Eight focused unit tests covering the full pure-data pipeline.
