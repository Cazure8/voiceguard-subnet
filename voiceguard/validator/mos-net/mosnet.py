import os
import numpy as np
from model import CNN_BLSTM
import utils

class MOSNet:
    """Wrapper for loading and running MOSNet model."""
    def __init__(self, pretrained_model_path="./pretrained/cnn_blstm.h5"):
        self.model_path = pretrained_model_path
        self.model = self._load_model()

    def _load_model(self):
        """Load the pretrained MOSNet model."""
        print("Loading MOSNet model weights...")
        MOSNet = CNN_BLSTM()
        model = MOSNet.build()
        model.load_weights(self.model_path)
        return model

    def predict(self, wavfile):
        """Predict the MOS score for a given wavfile."""
        if not os.path.isfile(wavfile):
            raise FileNotFoundError(f"The specified file {wavfile} does not exist.")

        # Generate spectrogram
        mag_sgram = utils.get_spectrograms(wavfile)
        timestep = mag_sgram.shape[0]
        mag_sgram = np.reshape(mag_sgram, (1, timestep, utils.SGRAM_DIM))

        # Make prediction
        Average_score, _ = self.model.predict(mag_sgram, verbose=0, batch_size=1)
        return Average_score[0][0]
