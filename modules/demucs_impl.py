import torch
from pathlib import Path
from demucs.apply import apply_model
from demucs import pretrained
from demucs.audio import prevent_clip
import torchaudio
import time


start_time = time.time()

model = pretrained.get_model('htdemucs') # hdemucs_mmi | htdemucs | htdemucs_ft | mdx
device = "cuda:0" if torch.cuda.is_available() else "mps"


def demucs_it(audio_path: str):
    data, sr = torchaudio.load(audio_path)
    data = torch.unsqueeze(data, 0)
    data.to(device)
    x = apply_model(model, data, device=device)[0]
    print(x.shape)
    other_stem = torch.zeros_like(x[0])
    vocals = torch.zeros_like(x[0])
    for name, source in zip(model.sources, x):
        if name != "vocals":
            other_stem += source
        else:
            vocals = source

    other_stem = prevent_clip(other_stem)
    bits_per_sample = 16
    encoding = 'PCM_S'
    vocals_tgt_dir = f"{Path(audio_path).parent.absolute().__str__()}/vocals.wav"
    no_vocals_tgt_dir = f"{Path(audio_path).parent.absolute().__str__()}/no_vocals.wav"
    torchaudio.save(vocals_tgt_dir, vocals, sample_rate=sr,encoding=encoding, bits_per_sample=bits_per_sample)
    torchaudio.save(no_vocals_tgt_dir, other_stem, sample_rate=sr,encoding=encoding, bits_per_sample=bits_per_sample)


if __name__ == "__main__":
    audio_path = './audios/JJy_AgTXwZ0_PART_.wav'
    demucs_it(audio_path)
    print("--- %s seconds ---" % (time.time() - start_time))
