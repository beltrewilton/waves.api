let wavesurfer;

const initwf = () => {
    wavesurfer = WaveSurfer.create({
      container: '#waveform',
      waveColor: '#4F4A85',
      progressColor: '#383351',
    //  url: 'https://127.0.0.1:8000/dltr?url=https://www.youtube.com/watch?v=WLL7bxoiOnw',
    })
}

document.addEventListener("DOMContentLoaded", async () => {
    initwf();
    const response = await fetch('https://127.0.0.1:8000/dltr?url=https://www.youtube.com/watch?v=WLL7bxoiOnw')
    const audiopath = await response.json()
    wavesurfer.load(audiopath)

    document.getElementById("button_play").addEventListener("click", () => {
        wavesurfer.playPause();
    });

})