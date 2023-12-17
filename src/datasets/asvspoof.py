import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import pandas as pd


def parse_protocol(path_to_protocol, audio_dir):
    protocol_df = pd.read_csv(path_to_protocol, sep=" ")
    data = pd.DataFrame({
        "path_to_audio": audio_dir + '/' + protocol_df.iloc[:, 1] + '.flac',
        "target": protocol_df.iloc[:, -1].replace(['bonafide', 'spoof'], [1, 0])
    })

    return data


class ASVSpoof(Dataset):
    def __init__(self, path_to_protocol, audio_dir, sample_size):
        self.path_to_protocol = Path(path_to_protocol)
        self.audio_dir = audio_dir
        self.data = parse_protocol(self.path_to_protocol, self.audio_dir)
        self.sample_size = sample_size
        
    def __getitem__(self, index):
        data_dict = self.data.iloc[index].to_dict()
        audio, sr = torchaudio.load(data_dict["path_to_audio"])
        data_dict["audio"] = audio
        return data_dict
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        processed_audio = []
        audio_paths = []
        targets = []
        for data_point in batch:
            audio = data_point["audio"]
            if audio.size(-1) < self.sample_size:
                padding = self.sample_size - audio.size(-1)
                audio = F.pad(audio, (0, padding), 'constant', 0)
            processed_audio.append(audio[..., :self.sample_size])
            audio_paths.append(data_point["path_to_audio"])
            targets.append(data_point["target"])

        return {
            "audio" : torch.cat(processed_audio).unsqueeze(1),
            "path_to_audio" : audio_paths,
            "target" : torch.tensor(targets)
        }