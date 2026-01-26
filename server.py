"""
Flask server for interactive spectrogram visualization.
Serves the viewer and provides an API for recomputing spectrograms with different sigma values.
"""

import os
import io
import base64
import glob
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from scipy.io import wavfile
from PIL import Image

from ifdv import ifdv

app = Flask(__name__, static_folder='.')

# Cache for loaded WAV data
wav_cache = {}

# Cache for computed spectrograms (keyed by parameters)
spectrogram_cache = {}

def get_cache_key(wav_name, sigmas, compute_type, fft_size, step_size, superres, lock_t, lock_f, log_offset, crop_f):
    """Generate a cache key from parameters."""
    sigma_str = ','.join(f'{s:.1f}' for s in sigmas)
    return f"{wav_name}|{sigma_str}|{compute_type}|{fft_size}|{step_size}|{superres}|{lock_t}|{lock_f}|{log_offset:.2f}|{crop_f:.1f}"

def load_wav(wav_path):
    """Load and normalize WAV file, with caching."""
    if wav_path in wav_cache:
        return wav_cache[wav_path]
    
    sampling_rate, data = wavfile.read(wav_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Normalize to float between -1 and 1
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128) / 128.0
    else:
        data = data.astype(np.float64)
    
    wav_cache[wav_path] = (sampling_rate, data)
    return sampling_rate, data


def compute_spectrogram(wav_name, sigmas, compute_type='ifdgram', fft_size=1024, step_size=72,
                        superres=1, lock_t=15, lock_f=5, log_offset=0.3, crop_f=1.0):
    """Compute spectrogram with given parameters and return as base64 PNG."""
    # Check cache first
    cache_key = get_cache_key(wav_name, sigmas, compute_type, fft_size, step_size, superres, lock_t, lock_f, log_offset, crop_f)
    if cache_key in spectrogram_cache:
        print(f"  Cache hit: {wav_name} ({compute_type})")
        return spectrogram_cache[cache_key], None
    
    zoom_t = 1  # Fixed at 1
    zoom_f = superres  # Superresolution is applied to frequency axis
    print(f"  Computing: {wav_name} ({compute_type}) σ={sigmas} fft={fft_size} step={step_size} superres={superres} lock=({lock_t},{lock_f}) logoff={log_offset}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(script_dir, wav_name)
    
    if not os.path.exists(wav_path):
        return None, f"WAV file not found: {wav_name}"
    
    sampling_rate, data = load_wav(wav_path)
    
    # Parameters
    n = fft_size
    overlap = n - step_size
    tl = lock_t
    fl = lock_f
    
    # Limit length for WebGL
    MAX_IMAGE_WIDTH = 16384
    step = n - overlap
    max_samples = (MAX_IMAGE_WIDTH - 1) * step + n
    
    if len(data) > max_samples:
        data = data[:len(data) // 2]
    
    # Compute for each sigma
    results = []
    for sigma in sigmas:
        ifdgram, sonogram, dx = ifdv(data, sampling_rate, n, overlap, sigma, zoom_t, zoom_f, tl, fl)
        if compute_type == 'ifdgram':
            results.append(ifdgram)
        else:
            results.append(np.abs(sonogram))
    
    # Crop to show only low frequency portion (crop_f=0.5 means bottom half, crop_f=1.0 means full)
    freq_bins = results[0].shape[0]
    keep_bins = int(freq_bins * crop_f)
    crop_start = freq_bins - keep_bins  # Start from high freq, keep low freq portion
    results = [img[crop_start:] for img in results]
    
    # Normalize each channel with log transform
    def normalize(img, offset):
        img = np.log(img + offset)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return np.zeros_like(img)
    
    channels = [normalize(img, log_offset) for img in results]
    rgb = np.stack(channels, axis=-1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    # Convert to base64 PNG
    img = Image.fromarray(rgb_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Store in cache
    spectrogram_cache[cache_key] = img_base64
    
    return img_base64, None


@app.route('/')
def index():
    return send_from_directory('.', 'viewer.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/wavfiles')
def list_wavfiles():
    """Return list of available WAV files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_files = glob.glob(os.path.join(script_dir, "*.wav"))
    wav_names = [os.path.basename(f) for f in sorted(wav_files)]
    return jsonify(wav_names)


@app.route('/api/compute', methods=['POST'])
def compute():
    """Compute spectrogram with given parameters."""
    data = request.get_json()
    
    wav_name = data.get('wav')
    sigmas = data.get('sigmas', [1.0, 2.0, 3.0])
    compute_type = data.get('type', 'ifdgram')  # 'ifdgram' or 'sonogram'
    fft_size = data.get('fft_size', 1024)
    step_size = data.get('step_size', 72)
    superres = data.get('superres', 1)
    lock_t = data.get('lock_t', 15)
    lock_f = data.get('lock_f', 5)
    log_offset = data.get('log_offset', 0.3)
    crop_f = data.get('crop_f', 1.0)
    
    if not wav_name:
        return jsonify({'error': 'No WAV file specified'}), 400
    
    # Validate sigmas
    try:
        sigmas = [float(s) for s in sigmas]
        if len(sigmas) != 3:
            return jsonify({'error': 'Must provide exactly 3 sigma values'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid sigma values'}), 400
    
    # Validate FFT and step size
    try:
        fft_size = int(fft_size)
        step_size = int(step_size)
        if fft_size not in [256, 512, 1024, 2048, 4096]:
            return jsonify({'error': 'FFT size must be 256, 512, 1024, 2048, or 4096'}), 400
        if step_size < 8 or step_size > fft_size:
            return jsonify({'error': f'Step size must be between 8 and {fft_size}'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid FFT or step size'}), 400
    
    # Validate superres, lock, log_offset, and crop parameters
    try:
        superres = int(superres)
        lock_t = int(lock_t)
        lock_f = int(lock_f)
        log_offset = float(log_offset)
        crop_f = float(crop_f)
        superres = max(1, min(10, superres))
        lock_t = max(1, min(50, lock_t))
        lock_f = max(1, min(50, lock_f))
        log_offset = max(0.01, min(10.0, log_offset))
        crop_f = max(0.1, min(1.0, crop_f))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid superres, lock, log_offset, or crop parameters'}), 400
    
    img_base64, error = compute_spectrogram(wav_name, sigmas, compute_type, fft_size, step_size,
                                            superres, lock_t, lock_f, log_offset, crop_f)
    
    if error:
        return jsonify({'error': error}), 400
    
    return jsonify({
        'image': img_base64,
        'sigmas': sigmas,
        'type': compute_type,
        'fft_size': fft_size,
        'step_size': step_size,
        'superres': superres,
        'lock_t': lock_t,
        'lock_f': lock_f,
        'crop_f': crop_f
    })


if __name__ == '__main__':
    print("Starting spectrogram server at http://localhost:8000")
    print("Open http://localhost:8000 in your browser")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
