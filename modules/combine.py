import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import librosa
from pathlib import Path
import re
import time


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

start_time = time.time()


def combine_chunks(audio_path: Path):
    key = audio_path.parent.name
    vocals = audio_path.parent
    volcals_path = vocals.absolute().__str__()
    vs = [v.name for v in sorted(vocals.rglob("*_synth_*.wav"))]
    vs.sort(key=natural_keys)
    combined = np.array([])
    silence = np.array([])
    sr = 0
    for k, v in enumerate(vs):
        if not combined.any():
            silence, _ = librosa.load(f'{volcals_path}/vocals-silence--{k}.wav')
            combined, sr = librosa.load(f'{volcals_path}/vocals_{k}_synth_a-0.3_b-0.2_df-10_em-1.wav')
            combined = np.concatenate((silence, combined), axis=0)
        else:
            silence, _ = librosa.load(f'{volcals_path}/vocals-silence--{k}.wav')
            y, _ = librosa.load(f'{volcals_path}/vocals_{k}_synth_a-0.3_b-0.2_df-10_em-1.wav')
            z = np.concatenate((silence, y), axis=0)
            combined = np.concatenate((combined, z), axis=0)
        print(v)

    write(f'{volcals_path}/combined.wav', sr, combined)
    no_vocals = AudioSegment.from_wav(f'{volcals_path}/no_vocals.wav')
    vocals = AudioSegment.from_wav(f'{volcals_path}/combined.wav')

    final_ = no_vocals.overlay(vocals)
    final_.export(f'{volcals_path}/final_sound.wav', format="wav")
