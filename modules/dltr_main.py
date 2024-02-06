import subprocess
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import scipy.signal as signal
import stable_whisper
import torch
import whisperx
from fastapi import APIRouter, Depends, HTTPException, status
from pydub import AudioSegment
from pytube import YouTube
from sqlalchemy.orm import Session
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration, pipeline
)

from dbms.database import get_db
from modules.combine import combine_chunks
from modules.demucs_impl import demucs_it
from modules.silero_based_chunk import chunk, split_audio


router = APIRouter(prefix='/dltr', tags=['dltr'])

gettrace = getattr(sys, 'gettrace', None)
# is debug mode :-) ?
# if gettrace():
#     path = os.getcwd() + '/dbms/query.sql'
#     print('Debugging :-* ')
# else:
#     path = os.getcwd() + '/dbms/query.sql'
#     print('Run normally.')
#
# query = Query(path)


output_path = "audios"
device = "cuda:0" if torch.cuda.is_available() else "mps"
model_name_hf = "openai/whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained(model_name_hf).to(device)
processor = WhisperProcessor.from_pretrained(model_name_hf)


def mp4_to_wav(input_mp4, output_wav):
    # Check if the input file exists
    if not os.path.exists(input_mp4):
        print(f"Error: Input file '{input_mp4}' not found.")
        return

    # command = [
    #     'ffmpeg',
    #     '-y',
    #     '-i', input_mp4,
    #     '-ss', '00:00:00',
    #     '-to', '00:00:09',
    #     '-acodec', 'pcm_s16le',
    #     '-ar', '16000',
    #     '-ac', '1',
    #     output_wav
    # ]

    command = [
        'ffmpeg',
        '-y',
        '-i', input_mp4,
        # '-ss', '00:01:28',
        # '-to', '00:05:57',
        '-acodec', 'pcm_s16le',
        '-hide_banner',
        '-loglevel', 'error',
        output_wav
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Error ffmpeg encontrado durante ejecucion, {result.stderr}")
        print(f"Conversion successful: {input_mp4} -> {output_wav},\n{result.stderr} {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


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
    # data = zscore(data)
    sampled = signal.resample(data, nums)
    return sampled


def tr_audio_st(sr, audio_reps):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = stable_whisper.load_faster_whisper("large-v3", device=device, compute_type="auto", num_workers=5,)
    model = stable_whisper.load_faster_whisper("large-v3", device=device,)
    result = model.transcribe(audio=audio_reps,)
    # result = model.transcribe(audio=audio_reps, beam_size=20, language='es', temperature=0, task='transcribe',
    #                                  word_timestamps=True, condition_on_previous_text=False, no_speech_threshold=0.1)

    return result


def tr_audio_pipe(sr, audio_reps):
    device = "cuda:0" if torch.cuda.is_available() else "mps"

    model_name_hf = "openai/whisper-large-v3"

    audio_reps = resample(audio_reps, sr, 16_000)

    # tokenizer = WhisperTokenizer.from_pretrained(model_name_hf, language="es", task="transcribe", verbose=True, beam_size=10)

    pipe = pipeline("automatic-speech-recognition", model=model_name_hf, device=device,)
    r = pipe(audio_reps, return_timestamps=True, chunk_length_s=30, stride_length_s=[4, 2], batch_size=8, generate_kwargs = {"language":"<|es|>","task": "transcribe"})
    return r


def tr_audio_x(sr, audio_reps):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    audio_reps = resample(audio_reps, sr, 16_000)
    model = whisperx.load_model("large-v3", device, compute_type='float32')
    result = model.transcribe(audio_reps, batch_size=8)

    return result


def tr_audio(sr, audio_reps, resample_=True): # numpy data
    if resample_:
        audio_reps = resample(audio_reps, sr, 16_000)

    # features and generate token ids
    input_features = processor.feature_extractor(audio_reps, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    # timestamp use.
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="Spanish", task="transcribe")
    # predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, return_timestamps=True,)
    # predicted_ids = model.generate(input_features, language="<|es|>",  task="transcribe", return_timestamps=True)
    predicted_ids = model.generate(input_features, language="<|es|>",  task="translate",)

    # decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # transcription = processor.tokenizer.decode(predicted_ids[0].cpu().numpy(), output_offsets=True)
    return transcription


def dl_audio(url: str):
    os.makedirs(output_path, exist_ok=True)
    ytb_key, pathfile = "", ""
    if "?" in url:
        ytb_key = url.split("?")[1]
        ytb_key = ytb_key.replace("v=", "") if "v=" in ytb_key else ytb_key
        os.makedirs(f"{output_path}/{ytb_key}", exist_ok=True)
        pathfile = f"{output_path}/{ytb_key}/{ytb_key[2:]}.mp4"
    else:
        ytb_key = url.split("shorts/")[1]
        os.makedirs(f"{output_path}/{ytb_key}", exist_ok=True)
        pathfile = f"{output_path}/{ytb_key}/{ytb_key}.mp4"

    if not os.path.exists(pathfile):
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=output_path)
        os.rename(out_file, pathfile)

    audio = AudioSegment.from_file(pathfile, format="mp4")
    # audio_base64 = base64.b64encode(audio.export(format="wav").read())
    # return audio_base64.decode("utf-8")
    # static_synth_url = f"https://127.0.0.1:8000/{pathfile.replace('.mp4', '_PART__synth.wav')}"
    static_url = f"https://127.0.0.1:8000/{pathfile}"
    return static_url, pathfile, audio.duration_seconds, audio.frame_rate, audio.channels


def get_audio_info(pathfile):
    audio = AudioSegment.from_file(pathfile, format="wav")
    return pathfile, audio.duration_seconds, audio.frame_rate, audio.channels


def synth_req(audio_path: str, text: str, alpha: float = 0.3, beta: float = 0.2) -> dict:
    data = {
        "audio_path": audio_path,
        "text": text,
        "alpha":  alpha,
        "beta":  beta,
    }
    url = "https://127.0.0.1:8060/tts/synth"
    public_pem = os.getcwd() + '/certs/public.crt'
    key_pem = os.getcwd() + '/certs/key.pem'
    r = requests.post(url=url, data=json.dumps(data), verify=False)
    return r.json()


def tr_chunks(audio_path: Path, cuts: dict) -> dict:
    key = audio_path.parent.name
    vocals = audio_path.parent
    trs = cuts.copy()
    for i, v in enumerate(sorted(vocals.rglob("vocals_*.wav"))):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f"translate {v}")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        data, sr = librosa.load(v, sr=16_000, mono=True)  # HACE NORMALIZACION
        transcript = tr_audio(sr, data, resample_=False)
        trs[i]['path'] = v.absolute().__str__()
        trs[i]['transcript'] = transcript[0]
    return trs


def synth_chunks(trs: dict):
    for i in trs.items():
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'synthetize {i[1]["path"]}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        r = synth_req(audio_path=i[1]['path'], text=i[1]['transcript'])
        print(r)


def mask_seq():
    pass


@router.get("/",)
async def get_audio(url: str, db: Session = Depends(get_db)):
    start_time = time.time()
    try:
        static_url, mp4_file, duration_seconds, frame_rate, channels = dl_audio(url) # dl & calc el wav nativo
        output_wav = mp4_file.replace(".mp4", ".wav")
        mp4_to_wav(input_mp4=mp4_file, output_wav=output_wav)
        audio_path = Path(output_wav)

        demucs_it(audio_path.absolute().__str__())
        vocals = f"{audio_path.parent.absolute().__str__()}/vocals.wav"

        cuts = chunk(vocals)
        split_audio(cuts, vocals)
        trs = tr_chunks(audio_path, cuts)
        synth_chunks(trs)
        combine_chunks(audio_path)

        synth_wav_file = f"{audio_path.parent.absolute().__str__()}/final_sound.wav"
        key = audio_path.parent.name
        synth_static_url = f"https://127.0.0.1:8000/audios/{key}/final_sound.wav"
        synth_pathfile, synth_duration_seconds, synth_frame_rate, synth_channels = get_audio_info(synth_wav_file)

        # PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -m demucs.separate -n htdemucs audios/_K_BlmXSlrw_PART_.wav  --two-stems=vocals -d mps
        #
        #TODO: hacer componente para mostrar lang original y lang synthetico
        #TODO: entender styletts inference
        #TODO: (algoritmo) usar transcript con timestamp con uteraciones de 15-25 segundos en promedio:
        #TODO: (algoritmo) por cada uteracion par [audio+transcript] aplicarle un par [refembedding+clone]
        #TODO: identificar silencios largas (>1s); agregar 'audio silencio'
        #TODO: hacer merge de todo los audios generados.
        #TODO: nice2have sacar la musica y mergear al final.
        #TODO: integrar a base de datos (detalle coming soon)

        # ---- HEAVY AREA ----
        #TODO: entender styletts train
        #TODO: entender espeak
        #TODO: profundizar phonemes, posible consulta del libro speech
        #TODO: deep lecturac en foros ....
        #TODO: entender el mel spectrogram
        #TODO: conocer mas sobre Fundamental frequency (F0)

        print("--- %s seconds ---" % (time.time() - start_time))

        return {
            'native': {
                'static_url': static_url,
                'pathfile': mp4_file,
                'duration_seconds': duration_seconds,
                'frame_rate': frame_rate,
                'channels': channels,
            },
            'synth': {
                'static_url': synth_static_url,
                'pathfile': synth_pathfile,
                'duration_seconds': synth_duration_seconds,
                'frame_rate': synth_frame_rate,
                'channels': synth_channels,
            }
        }
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


if __name__ == "__main__":
    start_time = time.time()

    key = "q3-2U2YxlWk_PART_"
    # key = "JJy_AgTXwZ0_PART_"
    vocals = Path(f'../separated/htdemucs/{key}')
    for v in sorted(vocals.rglob("vocals_*.wav")):
        # sr, data = wavfile.read(v)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(v)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        data, sr = librosa.load(v, sr=16_000, mono=True) # HACE NORMALIZACION
        transcript = tr_audio(sr, data, resample_=False)
        print(transcript)
        r = synth_req(audio_path=v.absolute().__str__(), text=transcript[0])
        print(r)

    print("--- %s seconds ---" % (time.time() - start_time))



    # key = "JJy_AgTXwZ0_PART_"
    # data, sr = librosa.load(f'../separated/htdemucs/{key}/vocals_2.wav', sr=16_000, mono=True)  # HACE NORMALIZACION
    # transcript = tr_audio(sr, data, resample_=False)
    #
    # import itertools
    #
    # alphas = [0.0,  0.2,  0.4,  0.6,  0.8,  1]
    # betas = [0.0,  0.2,  0.4,  0.6,  0.8,  1]
    # # ref = "lN_dkoxbAlw_PART_.wav"
    # ref = "vocals_2.wav"
    # # transcript = "And check this curious fact, it blew my mind In all the territory of Greenland there are only 4 traffic lights."
    #
    # for j, k in list(itertools.combinations(range(len(alphas)), r=2)):
    #     print(alphas[j], betas[k])
    #     r = synth_req(ref, key, transcript[0], alpha=alphas[j], beta=betas[k])
    #     print(r['synth_name'], r['response'])
