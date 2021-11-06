import audio as Audio
import librosa
import numpy as np
import yaml

class Preprocessor:
    def __init__(self, config):
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.STFT = Audio.stft.TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )

    def get_file_mel(self, in_path, out_path):
        pass

    def get_single_mel(self, in_wav_path):
        wav, _ = librosa.load(in_wav_path, self.sampling_rate)
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        
        return mel_spectrogram.T
        

if __name__ == "__main__":
    config = yaml.load(open('./config.yaml', "r"), Loader=yaml.FullLoader)
    
    preprocessor = Preprocessor(config)
    mel = preprocessor.get_single_mel('test.wav')
    # out_mel_shape: [frame_length, 80]
    
    np.save('test.npy', mel)