import os
import urllib.request
import soundfile as sf
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audio", "atmosphere")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full duration audio tracks with loop smoothing
TRACKS = {
    "atmosphere_rain.mp3": {
        "url": "https://raw.githubusercontent.com/bradtraversy/ambient-sound-mixer/master/audio/rain.mp3",
        "name": "Rain Ambience",
        "gain_db": -1.0,
        "max_duration_sec": 180,  # loop 108s to ~180s (3 mins)
    },
    "atmosphere_forest.mp3": {
        "url": "https://raw.githubusercontent.com/bradtraversy/ambient-sound-mixer/master/audio/wind.mp3",
        "name": "Forest Breeze & Wind",
        "gain_db": -1.0,
        "max_duration_sec": 290,  # full ~4.8 mins
    },
    "atmosphere_cafe.mp3": {
        "url": "https://raw.githubusercontent.com/bradtraversy/ambient-sound-mixer/master/audio/cafe.mp3",
        "name": "Cafe Atmosphere",
        "gain_db": -2.0,
        "max_duration_sec": 300,  # 5 full mins (original is 10 mins)
    },
    "atmosphere_horror.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c2/John_Bartmann_-_dark-basement-master.ogg",
        "name": "Horror - Dark Basement (John Bartmann CC0)",
        "gain_db": 0.0,
        "max_duration_sec": 240,  # full 4 mins
    },
    "atmosphere_action.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/John_Bartmann_-_endgame-master.ogg",
        "name": "Action - Endgame (John Bartmann CC0)",
        "gain_db": 0.0,
        "max_duration_sec": 240,  # full 4 mins
    },
    "atmosphere_cyberpunk.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/51/John_Bartmann_-_dystopia-master.ogg",
        "name": "Cyberpunk - Dystopia (John Bartmann CC0)",
        "gain_db": 0.0,
        "max_duration_sec": 240,  # full 4 mins
    },
    "atmosphere_mystery.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/30/John_Bartmann_-_broken-suspense-master.ogg",
        "name": "Mystery - Suspense (John Bartmann CC0)",
        "gain_db": 0.0,
        "max_duration_sec": 240,  # full 4 mins
    },
    "atmosphere_historical.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/28/John_Bartmann_-_aetherbells-master.ogg",
        "name": "Historical - Aether Bells (John Bartmann CC0)",
        "gain_db": 0.0,
        "max_duration_sec": 240,  # full 4 mins
    },
}

def make_seamless_loop(audio_data, sr, target_sec=None):
    crossfade_len = int(5.0 * sr)  # 5 second equal-power crossfade
    
    if target_sec is not None and len(audio_data) < int(target_sec * sr):
        # Tile audio if shorter than target
        repeats = int(np.ceil((target_sec * sr) / len(audio_data)))
        audio_data = np.tile(audio_data, (repeats, 1))
        
    if target_sec is not None:
        target_len = int(target_sec * sr)
        audio_data = audio_data[:target_len + crossfade_len]

    if len(audio_data) > crossfade_len * 2:
        seg = audio_data
        body_len = len(seg) - crossfade_len
        head = seg[:crossfade_len]
        body = seg[crossfade_len:body_len]
        tail = seg[body_len:body_len + crossfade_len]

        fade_in = np.linspace(0.0, 1.0, crossfade_len).reshape(-1, 1)
        fade_out = 1.0 - fade_in

        crossfade_head = tail * np.sqrt(fade_out) + head * np.sqrt(fade_in)
        return np.vstack([crossfade_head, body])
    return audio_data

def process_full_audio():
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_raw_full")
    os.makedirs(temp_dir, exist_ok=True)

    for filename, meta in TRACKS.items():
        print(f"\nProcessing full-length {filename} ({meta['name']})...")
        ext = ".ogg" if ".ogg" in meta["url"] else ".mp3"
        temp_input = os.path.join(temp_dir, f"temp_{filename}{ext}")

        print(f"Downloading from {meta['url']}...")
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "ShinKuBot/1.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
        with open(temp_input, "wb") as f:
            f.write(content)

        data, sr = sf.read(temp_input)
        if len(data.shape) == 1:
            data = np.column_stack([data, data])

        processed = make_seamless_loop(data, sr, meta.get("max_duration_sec"))

        # Normalize RMS volume to -20 dBFS
        rms = np.sqrt(np.mean(processed**2))
        if rms > 0:
            target_rms = 10 ** ((-20.0 + meta["gain_db"]) / 20.0)
            processed = processed * (target_rms / rms)

        # Peak limiter
        peak = np.max(np.abs(processed))
        if peak > 0.95:
            processed = processed * (0.95 / peak)

        out_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(out_path, processed, sr, format="MP3")
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        duration_min = (len(processed) / sr) / 60
        print(f"Done: {filename} -> {duration_min:.2f} mins, {file_size_mb:.2f} MB")

        try:
            os.remove(temp_input)
        except Exception:
            pass

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    print("\nAll full-length audio tracks processed and saved!")

if __name__ == "__main__":
    process_full_audio()
