import os
import math
import numpy as np
import soundfile as sf

SR = 44100
DURATION = 60.0  # 60 seconds
N_SAMPLES = int(SR * DURATION)
OUT_DIR = os.path.join(os.path.dirname(__file__), "audio", "atmosphere")
os.makedirs(OUT_DIR, exist_ok=True)

def make_seamless_loop(audio, fade_len=SR * 3):
    # Crossfade start and end by fade_len samples
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = 1.0 - fade_in
    if audio.ndim == 2:
        fade_in = fade_in[:, None]
        fade_out = fade_out[:, None]
    
    # Overlap tail onto head
    head = audio[:fade_len] * fade_in + audio[-fade_len:] * fade_out
    result = audio.copy()
    result[:fade_len] = head
    return result[:-fade_len]

def normalize(audio, target_db=-18.0):
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    target_rms = 10.0 ** (target_db / 20.0)
    audio = audio * (target_rms / rms)
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio

def generate_rain():
    print("Generating atmosphere_rain.mp3...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Background steady pinkish rain wash
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    # Cumulative sum to approximate brownian/pink noise
    pink = np.cumsum(white, axis=0)
    pink = pink - np.mean(pink, axis=0)
    pink = pink / np.max(np.abs(pink))
    
    # Modulation (wind gust in rain)
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.05 * t)[:, None]
    rain_wash = pink * mod
    
    # Randomized individual droplet impacts
    n_drops = int(DURATION * 180)
    drop_indices = np.random.randint(0, N_SAMPLES - 2000, n_drops)
    drops = np.zeros((N_SAMPLES, 2))
    for idx in drop_indices:
        dur = np.random.randint(400, 1500)
        t_d = np.linspace(0, dur / SR, dur)
        freq = np.random.uniform(1800, 4200)
        env = np.exp(-t_d * np.random.uniform(40, 120))
        d_wave = np.sin(2 * np.pi * freq * t_d) * env
        pan = np.random.uniform(0.1, 0.9)
        drops[idx:idx+dur, 0] += d_wave * (1.0 - pan)
        drops[idx:idx+dur, 1] += d_wave * pan
        
    combined = rain_wash * 0.7 + drops * 0.3
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-18.0)

def generate_forest():
    print("Generating atmosphere_forest.mp3 (Pure Tranquil Canopy & Leaves)...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Deep organic wind turbulence
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    wind = np.cumsum(white, axis=0)
    wind = wind - np.mean(wind, axis=0)
    wind = wind / np.max(np.abs(wind))
    
    # Slow dynamic forest breeze modulation
    breeze = (0.55 + 0.3 * np.sin(2 * np.pi * 0.025 * t) + 0.15 * np.sin(2 * np.pi * 0.07 * t))[:, None]
    forest_wind = wind * breeze
    
    # Soft rustling leaves (filtered high-mid bandpass noise modulated by wind)
    white2 = np.random.normal(0, 1, (N_SAMPLES, 2))
    leaves_wash = np.cumsum(white2, axis=0)
    leaves_wash = leaves_wash - np.mean(leaves_wash, axis=0)
    leaves_wash = leaves_wash / np.max(np.abs(leaves_wash))
    leaves = leaves_wash * (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 0.05 * t)))[:, None]

    combined = forest_wind * 0.75 + leaves * 0.25
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-20.0)

def generate_horror():
    print("Generating atmosphere_horror.mp3...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Deep detuned eerie sub-bass drone (55Hz and 55.8Hz for binaural beating)
    drone_l = np.sin(2 * np.pi * 55.0 * t) + 0.5 * np.sin(2 * np.pi * 110.0 * t) + 0.3 * np.sin(2 * np.pi * 165.0 * t)
    drone_r = np.sin(2 * np.pi * 55.7 * t) + 0.5 * np.sin(2 * np.pi * 110.5 * t) + 0.3 * np.sin(2 * np.pi * 164.3 * t)
    drone = np.column_stack([drone_l, drone_r])
    
    # Dark resonant minor overtone swell (Eb - 155.56Hz)
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * 0.04 * t)[:, None]
    eerie_note = np.sin(2 * np.pi * 155.56 * t)[:, None] * swell * 0.3
    
    # Cold cavernous background wind
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    rumble = np.cumsum(np.cumsum(white, axis=0), axis=0)
    rumble = rumble - np.mean(rumble, axis=0)
    rumble = rumble / np.max(np.abs(rumble))
    
    combined = drone * 0.4 + eerie_note * 0.3 + rumble * 0.3
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-18.0)

def generate_mystery():
    print("Generating atmosphere_mystery.mp3...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Wet city night hum
    hum_l = np.sin(2 * np.pi * 60.0 * t) + 0.4 * np.sin(2 * np.pi * 120.0 * t)
    hum_r = np.sin(2 * np.pi * 60.5 * t) + 0.4 * np.sin(2 * np.pi * 120.3 * t)
    hum = np.column_stack([hum_l, hum_r])
    
    # Soft window rain
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    soft_rain = np.cumsum(white, axis=0)
    soft_rain = soft_rain - np.mean(soft_rain, axis=0)
    soft_rain = soft_rain / np.max(np.abs(soft_rain))
    
    # Distant resonant chord swell (minor 7th)
    swell = (0.5 + 0.5 * np.sin(2 * np.pi * 0.06 * t))[:, None]
    chord = (np.sin(2 * np.pi * 130.81 * t) + np.sin(2 * np.pi * 155.56 * t) + np.sin(2 * np.pi * 196.00 * t))[:, None] * swell * 0.2
    
    combined = hum * 0.35 + soft_rain * 0.45 + chord * 0.2
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-20.0)

def generate_cyberpunk():
    print("Generating atmosphere_cyberpunk.mp3...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Warm analog synth pad drone with chorus (65.4Hz C2)
    f0 = 65.41
    synth_l = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * (f0*2) * t) + 0.25 * np.sin(2 * np.pi * (f0*3) * t)
    synth_r = np.sin(2 * np.pi * (f0+0.4) * t) + 0.4 * np.sin(2 * np.pi * (f0*2+0.8) * t) + 0.25 * np.sin(2 * np.pi * (f0*3+1.2) * t)
    
    # Filter sweep modulation
    lfo = (0.5 + 0.4 * np.sin(2 * np.pi * 0.08 * t))[:, None]
    synth = np.column_stack([synth_l, synth_r]) * lfo
    
    # Subtle electronic room air tone
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    room = np.cumsum(white, axis=0)
    room = room - np.mean(room, axis=0)
    room = room / np.max(np.abs(room))
    
    combined = synth * 0.6 + room * 0.4
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-19.0)

def generate_cafe():
    print("Generating atmosphere_cafe.mp3...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Low-pass room murmur wash
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    wash = np.cumsum(np.cumsum(white, axis=0), axis=0)
    wash = wash - np.mean(wash, axis=0)
    wash = wash / np.max(np.abs(wash))
    
    # Subtle ceramic cup taps and clinks
    clinks = np.zeros((N_SAMPLES, 2))
    n_clinks = int(DURATION * 16)
    indices = np.random.randint(SR * 2, N_SAMPLES - SR * 2, n_clinks)
    for idx in indices:
        dur = int(SR * np.random.uniform(0.08, 0.2))
        t_k = np.linspace(0, dur / SR, dur)
        freq = np.random.uniform(2200, 3600)
        env = np.exp(-t_k * np.random.uniform(35, 70))
        tone = np.sin(2 * np.pi * freq * t_k) * env * 0.25
        pan = np.random.uniform(0.2, 0.8)
        clinks[idx:idx+dur, 0] += tone * (1.0 - pan)
        clinks[idx:idx+dur, 1] += tone * pan
        
    combined = wash * 0.75 + clinks * 0.25
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-20.0)

def generate_historical():
    print("Generating atmosphere_historical.mp3...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Bamboo mountain wind
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    wind = np.cumsum(white, axis=0)
    wind = wind - np.mean(wind, axis=0)
    wind = wind / np.max(np.abs(wind))
    wind_mod = (0.5 + 0.3 * np.sin(2 * np.pi * 0.04 * t))[:, None]
    bamboo_wind = wind * wind_mod
    
    # Resonant Japanese/oriental pentatonic wind chimes (D, F, G, A, C)
    chimes = np.zeros((N_SAMPLES, 2))
    pentatonic_freqs = [587.33, 698.46, 783.99, 880.00, 1046.50, 1174.66]
    n_rings = int(DURATION * 20)
    ring_indices = np.random.randint(SR, N_SAMPLES - SR * 2, n_rings)
    for idx in ring_indices:
        f = np.random.choice(pentatonic_freqs)
        dur = int(SR * np.random.uniform(0.8, 2.2))
        dur = min(dur, N_SAMPLES - idx)
        if dur <= 0:
            continue
        t_r = np.linspace(0, dur / SR, dur)
        env = np.exp(-t_r * np.random.uniform(2.5, 4.5))
        chime = (np.sin(2 * np.pi * f * t_r) + 0.3 * np.sin(2 * np.pi * f * 2.02 * t_r)) * env * 0.25
        pan = np.random.uniform(0.2, 0.8)
        chimes[idx:idx+dur, 0] += chime[:dur] * (1.0 - pan)
        chimes[idx:idx+dur, 1] += chime[:dur] * pan

    combined = bamboo_wind * 0.7 + chimes * 0.3
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-20.0)

def generate_action():
    print("Generating atmosphere_action.mp3 (Cinematic Battle Pulse, Campfire & Winds)...")
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Deep roaring gusting winds
    white = np.random.normal(0, 1, (N_SAMPLES, 2))
    wind = np.cumsum(white, axis=0)
    wind = wind - np.mean(wind, axis=0)
    wind = wind / np.max(np.abs(wind))
    gust = (0.55 + 0.45 * np.sin(2 * np.pi * 0.05 * t))[:, None]
    storm_wind = wind * gust
    
    # Rhythmic low-frequency battle drum pulse (steady 60 bpm pulse)
    bpm = 60.0
    beat_interval = int(SR * (60.0 / bpm))
    battle_pulse = np.zeros((N_SAMPLES, 2))
    for beat_start in range(0, N_SAMPLES - SR, beat_interval):
        hit_len = int(SR * 0.45)
        t_hit = np.linspace(0, 0.45, hit_len)
        hit_wave = (np.sin(2 * np.pi * 50.0 * t_hit) + 0.4 * np.sin(2 * np.pi * 100.0 * t_hit)) * np.exp(-t_hit * 8.0)
        battle_pulse[beat_start:beat_start+hit_len, 0] += hit_wave * 0.45
        battle_pulse[beat_start:beat_start+hit_len, 1] += hit_wave * 0.45

    # Campfire crackle impulses
    crackle = np.zeros((N_SAMPLES, 2))
    n_pops = int(DURATION * 100)
    pop_indices = np.random.randint(0, N_SAMPLES - 3000, n_pops)
    for idx in pop_indices:
        dur = np.random.randint(200, 800)
        dur = min(dur, N_SAMPLES - idx)
        if dur <= 0:
            continue
        t_p = np.linspace(0, dur / SR, dur)
        pop_w = np.sin(2 * np.pi * np.random.uniform(800, 2400) * t_p) * np.exp(-t_p * 80) * 0.35
        pan = np.random.uniform(0.1, 0.9)
        crackle[idx:idx+dur, 0] += pop_w[:dur] * (1.0 - pan)
        crackle[idx:idx+dur, 1] += pop_w[:dur] * pan
        
    combined = storm_wind * 0.45 + battle_pulse * 0.35 + crackle * 0.20
    combined = make_seamless_loop(combined)
    return normalize(combined, target_db=-18.0)

generators = {
    "atmosphere_rain.mp3": generate_rain,
    "atmosphere_forest.mp3": generate_forest,
    "atmosphere_horror.mp3": generate_horror,
    "atmosphere_mystery.mp3": generate_mystery,
    "atmosphere_cyberpunk.mp3": generate_cyberpunk,
    "atmosphere_cafe.mp3": generate_cafe,
    "atmosphere_historical.mp3": generate_historical,
    "atmosphere_action.mp3": generate_action,
}

for filename, gen_func in generators.items():
    audio_data = gen_func()
    out_path = os.path.join(OUT_DIR, filename)
    sf.write(out_path, audio_data, SR, format="MP3")
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

print("All 8 ambient soundscapes generated successfully!")
