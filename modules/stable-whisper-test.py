# audio 16K
# func pasar path, [rango s - e "10 segundos de audio"] y retornar numpy
# inferir con
# buscar el argmax del silencio las sentences (diveididas por puntos )
#
import torch
import numpy as np
import stable_whisper
from scipy.io import wavfile
import scipy.signal as signal
from pydub import AudioSegment


def stereo_to_mono(data):
    """Convert stereo audio to mono by averaging the samples"""
    mono_audio = np.mean(data, axis=1, dtype=data.dtype)
    return mono_audio


def resample(data, orig_sr, target_sr):
    """resample down/up a data audio"""
    ratio = orig_sr / target_sr
    nums = int(len(data) / ratio)
    if len(data.shape) > 1:
        data = stereo_to_mono(data)
    sampled = signal.resample(data, nums)
    return sampled

SAMPLE_RATE = 16_000
vocals = '../separated/htdemucs/OjTfhzVbY7g_PART_/vocals16k.wav'
# audio = AudioSegment.from_wav(vocals,)
# s = 10.5 * 1000
# e = 23.76 * 1000
# audio = audio[s:e]
# audio.export("chunk_lab.wav", format="wav")

sr, audio = wavfile.read(vocals)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = stable_whisper.load_model("large-v3",
                                  device=device,
                                  in_memory=False,
                                  cpu_preload=True,
                                  dq=False,
                                  )

result = model.transcribe(audio=audio,
                          verbose=True,
                          word_timestamps=False,
                          suppress_silence=True,
                          )

print(result)