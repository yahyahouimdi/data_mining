# 1D CNN

Run `notebooks/05_cnn_1d.ipynb` locally or in Colab with PyTorch installed. The notebook uses sequence windows, early stopping, class weighting, and saves a `.pt` checkpoint plus training history.

For a specific exported dataset, load that folder's `whole.csv`, use only its train/development rows for fitting, preserve recording boundaries, and save to `models/<dataset>/cnn_1d.pt`.
