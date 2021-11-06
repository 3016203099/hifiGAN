import torch
import numpy as np
import json
import hifigan
import yaml
from scipy.io import wavfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hifiGAN:
    def __init__(self, device):
        self.vocoder = self.get_vocoder(device)

    def get_vocoder(self, device):
        with open("./hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load("./hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

        return vocoder

    def vocoder_infer(self, mels, preprocess_config):
        with torch.no_grad():
            wavs = self.vocoder(mels).squeeze(1)
        wavs = (
            wavs.cpu().numpy()
            * preprocess_config["audio"]["max_wav_value"]
        ).astype("int16")
        wavs = [wav for wav in wavs]

        return wavs

if __name__ == "__main__":
    config = yaml.load(open('./config.yaml', "r"), Loader=yaml.FullLoader)
    
    mels = np.array([np.load('./test.npy', allow_pickle=False)])
    mels = torch.from_numpy(mels).float().to(device)
    mels = mels.transpose(1,2)

    vocoder = hifiGAN(device)
    wavs = vocoder.vocoder_infer(mels, config)
    
    wavfile.write("test_synthesize.wav", config["audio"]["sampling_rate"], wavs[0])