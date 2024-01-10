from fastapi import FastAPI, File, HTTPException, Query, Depends
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from io import BytesIO
from sqlalchemy.orm import Session
from dbms.database import get_db, get_cursor

router = APIRouter(prefix='/stream', tags=['stream'])


def seek_audio(input_audio: AudioSegment, start_time: int) -> AudioSegment:
    # Seek to the specified start time
    seek_position = start_time * 1000  # Convert to milliseconds
    return input_audio[seek_position:]


@router.get('/')
async def hello():
    return {'hello': 'All ready.'}


@router.get("/db-dummy")
async def dummy(db: Session = Depends(get_db)):
    cur = get_cursor(db)
    sql_raw = "SELECT uname FROM users;"
    cur.execute(sql_raw)
    resp = cur.fetchall()
    print(resp)


@router.get('/audio-info')
async def info():
    try:
        audio = AudioSegment.from_wav('audio.wav')
        return {
            'audio_length': audio.duration_seconds,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio: {str(e)}")


@router.get("/seek-audio")
async def seek_audio_route(start_time: int = Query(..., ge=0)):
    try:
        # Load the audio from the received file bytes
        audio = AudioSegment.from_wav('audio.wav')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio: {str(e)}")

    # Seek the audio to the specified start time
    seeked_audio = seek_audio(audio, start_time)

    # Convert the audio to bytes
    audio_bytes = seeked_audio.export(format="wav").read()

    # Return a StreamingResponse with the audio content
    return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")
