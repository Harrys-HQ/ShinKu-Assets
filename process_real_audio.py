import os
import urllib.request
import soundfile as sf
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audio", "atmosphere")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACKS = {
    "atmosphere_rain.mp3": {
        "url": "https://raw.githubusercontent.com/bradtraversy/ambient-sound-mixer/master/audio/rain.mp3",
        "name": "Rain Ambience",
        "target_sec": 75,
        "gain_db": -1.0,
    },
    "atmosphere_forest.mp3": {
        "url": "https://raw.githubusercontent.com/bradtraversy/ambient-sound-mixer/master/audio/wind.mp3",
        "name": "Forest Breeze & Wind",
        "target_sec": 75,
        "gain_db": -1.0,
    },
    "atmosphere_cafe.mp3": {
        "url": "https://raw.githubusercontent.com/bradtraversy/ambient-sound-mixer/master/audio/cafe.mp3",
        "name": "Cafe Atmosphere",
        "target_sec": 75,
        "gain_db": -2.0,
    },
    "atmosphere_horror.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c2/John_Bartmann_-_dark-basement-master.ogg",
        "name": "Horror - Dark Basement (John Bartmann CC0)",
        "target_sec": 75,
        "gain_db": 0.0,
    },
    "atmosphere_action.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/John_Bartmann_-_endgame-master.ogg",
        "name": "Action - Endgame (John Bartmann CC0)",
        "target_sec": 75,
        "gain_db": 0.0,
    },
    "atmosphere_cyberpunk.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/51/John_Bartmann_-_dystopia-master.ogg",
        "name": "Cyberpunk - Dystopia (John Bartmann CC0)",
        "target_sec": 75,
        "gain_db": 0.0,
    },
    "atmosphere_mystery.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/30/John_Bartmann_-_broken-suspense-master.ogg",
        "name": "Mystery - Suspense (John Bartmann CC0)",
        "target_sec": 75,
        "gain_db": 0.0,
    },
    "atmosphere_historical.mp3": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/28/John_Bartmann_-_aetherbells-master.ogg",
        "name": "Historical - Aether Bells (John Bartmann CC0)",
        "target_sec": 75,
        "gain_db": 0.0,
    },
}

def download_and_process():
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_raw_audio")
    os.makedirs(temp_dir, exist_ok=True)

    for filename, meta in TRACKS.items():
        print(f"\nProcessing {filename} ({meta['name']})...")
        ext = ".ogg" if ".ogg" in meta["url"] else ".mp3"
        temp_input = os.path.join(temp_dir, f"temp_{filename}{ext}")

        print(f"Downloading from {meta['url']}...")
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "ShinKuBot/1.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
        with open(temp_input, "wb") as f:
            f.write(content)

        print(f"Reading audio data ({len(content)} bytes)...")
        data, sr = sf.read(temp_input)
        if len(data.shape) == 1:
            data = np.column_stack([data, data])

        target_len = int(meta["target_sec"] * sr)
        crossfade_len = int(4.0 * sr) # 4 second seamless crossfade

        if len(data) > target_len + crossfade_len:
            # Take target_len + crossfade_len from middle to avoid silence at start
            start_offset = min(int(3.0 * sr), len(data) - target_len - crossfade_len)
            seg = data[start_offset:start_offset + target_len + crossfade_len]
        else:
            seg = data

        if len(seg) > crossfade_len * 2:
            body_len = len(seg) - crossfade_len
            head = seg[:crossfade_len]
            body = seg[crossfade_len:body_len]
            tail = seg[body_len:body_len + crossfade_len]

            fade_in = np.linspace(0.0, 1.0, crossfade_len).reshape(-1, 1)
            fade_out = 1.0 - fade_in

            # Equal power crossfade
            crossfade_head = tail * np.sqrt(fade_out) + head * np.sqrt(fade_in)
            processed = np.vstack([crossfade_head, body])
        else:
            processed = seg

        # Normalize RMS volume to pleasant ambient reading level (-20 dBFS)
        rms = np.sqrt(np.mean(processed**2))
        if rms > 0:
            target_rms = 10 ** ((-20.0 + meta["gain_db"]) / 20.0)
            processed = processed * (target_rms / rms)

        # Peak limiter to avoid any digital clipping
        peak = np.max(np.abs(processed))
        if peak > 0.95:
            processed = processed * (0.95 / peak)

        out_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(out_path, processed, sr, format="MP3")
        file_size_kb = os.path.getsize(out_path) / 1024
        duration = len(processed) / sr
        print(f"Successfully generated {filename}: {duration:.1f}s, {file_size_kb:.1f} KB")

        try:
            os.remove(temp_input)
        except Exception:
            pass

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    print("\nAll 8 real ambient soundscape tracks have been generated and saved!")

if __name__ == "__main__":
    download_and_process()
