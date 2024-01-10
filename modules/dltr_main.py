from fastapi import APIRouter

import sys
import os
from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, Security, status, Header
from sqlalchemy.orm import Session
import requests
import json
from typing import Optional
from dbms.database import get_db
import query.user_query as user_query
import schemas.users as schemas
import modules.user_models as models
from query.user_query import Token
from query.user_query import create_access_token, logout_token
from query.user_query import ACCESS_TOKEN_EXPIRE_MINUTES, ACCESS_TOKEN_EXPIRE_SECONDS
from query.user_query import validate_permissions
from dbms.Query import Query

router = APIRouter(prefix='/dltr', tags=['dltr'])

gettrace = getattr(sys, 'gettrace', None)
# is debug mode :-) ?
if gettrace():
    path = os.getcwd() + '/dbms/query.sql'
    print('Debugging :-* ')
else:
    path = os.getcwd() + '/dbms/query.sql'
    print('Run normally.')

query = Query(path)

output_path = "audios"


def mp3_to_wav(input_mp4, output_wav):
    import subprocess
    import os

    # Check if the input file exists
    if not os.path.exists(input_mp4):
        print(f"Error: Input file '{input_mp4}' not found.")
        return

    command = [
        'ffmpeg',
        '-y',
        '-i', input_mp4,
        '-ss', '00:00:00',
        '-to', '00:00:09',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        output_wav
    ]

    command_cut = [
        'ffmpeg',
        '-y',
        '-i', input_mp4,
        '-ss', '00:00:00',
        '-to', '00:00:09',
        '-acodec', 'pcm_s16le',
        output_wav.replace(".wav", "_PART_.wav")
    ]

    try:
        subprocess.run(command, check=True)
        subprocess.run(command_cut, check=True)
        print(f"Conversion successful: {input_mp4} -> {output_wav}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def read_wav(path: str): # sr, data
    from scipy.io import wavfile

    sr, data = wavfile.read(path)

    return sr, data


def tr_audio(audio_reps): # numpy data
    import wave
    import numpy as np
    import torch
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration, pipeline, VitsModel, AutoTokenizer
    )

    device = "cuda:0" if torch.cuda.is_available() else "mps"

    model_name_hf = "openai/whisper-large-v3"

    processor = WhisperProcessor.from_pretrained(model_name_hf)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_hf).to(device)
    # timestamp use.
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    # features and generate token ids
    input_features = processor(audio_reps, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    # predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, return_timestamps=True,)
    predicted_ids = model.generate(input_features, return_timestamps=True, task="translate")

    # decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


def dl_audio(url: str):
    from pydub import AudioSegment
    import base64
    from pytube import YouTube
    import os

    os.makedirs(output_path, exist_ok=True)
    ytb_key, pathfile = "", ""
    if "?" in url:
        ytb_key = url.split("?")[1]
        pathfile = f"{output_path}/{ytb_key[2:]}.mp4"
    else:
        ytb_key = url.split("shorts/")[1]
        pathfile = f"{output_path}/{ytb_key}.mp4"

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


def tmp_func_audio_already_synth(pathfile):
    from pydub import AudioSegment

    # pathfile = pathfile.replace('.mp4', '_PART__synth_embedding_scale_1.wav')
    audio = AudioSegment.from_file(pathfile, format="wav")

    return  pathfile, audio.duration_seconds, audio.frame_rate, audio.channels


def synth_req(part: str, text: str, alpha: float = 0.3, beta: float = 0.2) -> dict:
    data = {
        "part": part,
        "text": text,
        "alpha":  alpha,
        "beta":  beta,
    }
    url = "https://127.0.0.1:8060/tts/synth"
    public_pem = os.getcwd() + '/certs/public.crt'
    key_pem = os.getcwd() + '/certs/key.pem'
    r = requests.post(url=url, data=json.dumps(data), verify=False)
    return r.json()


@router.get("/",)
async def get_audio(url: str, db: Session = Depends(get_db)):
    try:
        static_url, mp4_file, duration_seconds, frame_rate, channels = dl_audio(url) # dl & calc el wav nativo
        output_wav = mp4_file.replace(".mp4", ".wav")
        mp3_to_wav(input_mp4=mp4_file, output_wav=output_wav) # out: .wav & _PART_.wav
        sr, audio_resp = read_wav(output_wav)
        transcript = tr_audio(audio_resp)
        print(transcript)

        part = f"{mp4_file[7:-4]}_PART_.wav"
        # text = "The check in this very curious act, I got a headache In all the territory of Greenland there are only four traffic lights In the whole island, a man offered us even a few fock guards As a souvenir, they also have fock guards, bear guards A very peculiar light trade"
        r = synth_req(part, transcript[0])

        synth_wav_file = r['synth_wav_file']
        static_url_syth = f"https://127.0.0.1:8000/audios/{r['synth_name']}"
        xpathfile, xduration_seconds, xframe_rate, xchannels = tmp_func_audio_already_synth(synth_wav_file) # calcula el wav synth
        #TODO: lei foro
        #TODO: organizar variables, funciones y objetos en momoria
        #TODO: downsampling audio para whisper para no guardar ese wav
        #TODO: hacer componente para mostrar lang original y lang synthetico
        #TODO: entender styletts inference
        #TODO: preparar waves.styletts como servicio al que le envio un audioreferencia + transcript y
        #      me retorna un audio synthetizado con voz referenciada (cloned)
        #TODO: (algoritmo) usar transcript con timestamp con uteraciones de 15-25 segundos en promedio:
        #TODO: (algoritmo) por cada uteracion par [audio+transcript] aplicarle un par [refembedding+clone]
        #TODO: identificar silencios largas (>1s); agregar 'audio silencio'
        #TODO: hacer merge de todo los audios generados.
        #TODO: nice2have sacar la musica y mergear al final.
        #TODO: integrar a base de datos (detalle coming soon)
        #TODO: entender styletts train
        #TODO: entender espeak
        #TODO: profundizar phonemes, posible consulta del libro speech
        #TODO: deep lecturac en foros ....
        #TODO: entender el mel spectrogram
        #TODO: conocer mas sobre Fundamental frequency (F0)

        return {
            'native': {
                'static_url': static_url,
                'pathfile': mp4_file,
                'duration_seconds': duration_seconds,
                'frame_rate': frame_rate,
                'channels': channels,
            },
            'synth': {
                'static_url': static_url_syth,
                'pathfile': xpathfile,
                'duration_seconds': xduration_seconds,
                'frame_rate': xframe_rate,
                'channels': xchannels,
            }
        }
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


if __name__ == "__main__":
    import itertools

    alphas = [0.0,  0.2,  0.4,  0.6,  0.8,  1]
    betas = [0.0,  0.2,  0.4,  0.6,  0.8,  1]
    ref = "lN_dkoxbAlw_PART_.wav"
    transcript = "48 hours in the state of Santa Catarina I'm going now for an adventure where I will show you how incredible is the state of Santa Catarina"

    for j, k in list(itertools.combinations(range(len(alphas)), r=2)):
        print(alphas[j], betas[k])
        r = synth_req(ref, transcript, alpha=alphas[j], beta=betas[k])
        print(r['synth_name'], r['response'])
