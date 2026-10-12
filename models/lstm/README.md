# LSTM

Run `notebooks/06_lstm.ipynb` locally or in Colab with PyTorch installed. The notebook uses same-recording sequences, validation early stopping, class weighting, and saves a `.pt` checkpoint plus training history.

For a specific exported dataset, load that folder's `whole.csv`, never cross recording or split boundaries, and save to `models/<dataset>/lstm.pt`.
