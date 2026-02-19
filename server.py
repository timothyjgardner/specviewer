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
                        superres=1, lock_t=5, lock_f=5, log_offset=0.3, crop_f=1.0):
    """Compute spectrogram with given parameters and return as base64 PNG."""
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
    
    # Limit length for WebGL texture size
    MAX_IMAGE_WIDTH = 16384
    step = n - overlap
    max_samples = (MAX_IMAGE_WIDTH - 1) * step + n
    
    if len(data) > max_samples:
        print(f"  Truncating from {len(data)} to {max_samples} samples (step={step})")
        data = data[:max_samples]
    
    # Compute for each sigma
    results = []
    for sigma in sigmas:
        ifdgram, sonogram, dx, t_disp, f_disp, _, _ = ifdv(data, sampling_rate, n, overlap, sigma, zoom_t, zoom_f, tl, fl)
        if compute_type == 'ifdgram':
            results.append(ifdgram)
        elif compute_type == 'zeros' or compute_type == 'combined':
            # Inverse magnitude - zeros become bright
            mag = np.abs(sonogram)
            inv_mag = 1.0 / (mag + 1e-10)  # Avoid division by zero
            # Take only lower half of inv_mag (positive frequencies)
            inv_mag = inv_mag[:n//2, :]
            # Resize inv_mag to match ifdgram dimensions (which is scaled by zoom_f)
            if inv_mag.shape != ifdgram.shape:
                from scipy.ndimage import zoom as scipy_zoom
                scale_f = ifdgram.shape[0] / inv_mag.shape[0]
                scale_t = ifdgram.shape[1] / inv_mag.shape[1]
                inv_mag = scipy_zoom(inv_mag, (scale_f, scale_t), order=1)
            results.append((inv_mag, ifdgram))  # Store both for combined view
        elif compute_type == 'crossings':
            # Binary detection: mark where both |t_disp| and |f_disp| are small
            # Take only lower half (positive frequencies)
            t_d = t_disp[:n//2, :]
            f_d = f_disp[:n//2, :]
            # Resize to match ifdgram dimensions
            if t_d.shape != ifdgram.shape:
                from scipy.ndimage import zoom as scipy_zoom
                scale_f = ifdgram.shape[0] / t_d.shape[0]
                scale_t = ifdgram.shape[1] / t_d.shape[1]
                t_d = scipy_zoom(t_d, (scale_f, scale_t), order=1)
                f_d = scipy_zoom(f_d, (scale_f, scale_t), order=1)
            results.append((t_d, f_d, ifdgram))
        else:
            # Sonogram view - resample to match ifdgram dimensions for alignment
            sono_mag = np.abs(sonogram)
            sono_mag = sono_mag[:n//2, :]  # Keep only positive frequencies
            if sono_mag.shape != ifdgram.shape:
                from scipy.ndimage import zoom as scipy_zoom
                scale_f = ifdgram.shape[0] / sono_mag.shape[0]
                scale_t = ifdgram.shape[1] / sono_mag.shape[1]
                sono_mag = scipy_zoom(sono_mag, (scale_f, scale_t), order=1)
            results.append(sono_mag)
    
    # Crop to show only low frequency portion (crop_f=0.5 means bottom half, crop_f=1.0 means full)
    if compute_type in ['zeros', 'combined']:
        # Results are tuples of (inv_mag, ifdgram)
        first_item = results[0][0]
        freq_bins = first_item.shape[0]
        keep_bins = int(freq_bins * crop_f)
        crop_start = freq_bins - keep_bins
        results = [(inv[crop_start:], ifd[crop_start:]) for inv, ifd in results]
    elif compute_type == 'crossings':
        first_item = results[0][0]
        freq_bins = first_item.shape[0]
        keep_bins = int(freq_bins * crop_f)
        crop_start = freq_bins - keep_bins
        results = [(t[crop_start:], f[crop_start:], ifd[crop_start:]) 
                   for t, f, ifd in results]
    else:
        freq_bins = results[0].shape[0]
        keep_bins = int(freq_bins * crop_f)
        crop_start = freq_bins - keep_bins
        results = [img[crop_start:] for img in results]
    
    if compute_type == 'zeros':
        # For zeros view: use single channel with black→blue→cyan/white colormap
        # Use first sigma only for cleaner zeros pattern
        inv_mag, _ = results[0]
        
        # Clamp max inverse value to avoid blowing up at true zeros/padding
        max_inv = np.median(inv_mag) * 10  # Cap at 10x median
        zeros_norm = np.clip(inv_mag, 0, max_inv)
        
        # Log scale for better dynamic range
        zeros_norm = np.log(zeros_norm + 1)
        zeros_norm = zeros_norm / (zeros_norm.max() + 1e-10)
        
        # Apply black→blue→cyan→white colormap
        r = np.power(zeros_norm, 2.0) * 255
        g = np.power(zeros_norm, 1.2) * 255
        b = np.power(zeros_norm, 0.5) * 255
        
        rgb_uint8 = np.stack([r, g, b], axis=-1).astype(np.uint8)
    
    elif compute_type == 'combined':
        # Combined view: send zeros and ifdgram as separate channels for WebGL blending
        # R = zeros (normalized), G = ifdgram (normalized), B = 0
        inv_mag, ifdgram_data = results[0]
        
        # Process zeros - normalize
        max_inv = np.median(inv_mag) * 10
        zeros_norm = np.clip(inv_mag, 0, max_inv)
        zeros_norm = np.log(zeros_norm + 1)
        zeros_norm = zeros_norm / (zeros_norm.max() + 1e-10)
        
        # Process ifdgram - normalize with log
        ifd_norm = np.log(ifdgram_data + log_offset)
        ifd_norm = (ifd_norm - ifd_norm.min()) / (ifd_norm.max() - ifd_norm.min() + 1e-10)
        
        # Pack into RGB: R=zeros, G=ifdgram, B=0 (WebGL will apply colormaps)
        r = (zeros_norm * 255).astype(np.uint8)
        g = (ifd_norm * 255).astype(np.uint8)
        b = np.zeros_like(r)
        
        rgb_uint8 = np.stack([r, g, b], axis=-1)
    
    elif compute_type == 'crossings':
        # Contour-based detection: plot isolines where t_disp=0 and f_disp=0
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        t_d, f_d, ifdgram_data = results[0]
        
        # Create figure with exact pixel dimensions
        h, w = t_d.shape
        dpi = 100
        fig = Figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_axis_off()
        
        # Normalize ifdgram for background
        ifd_norm = np.log(ifdgram_data + 0.3)
        ifd_norm = (ifd_norm - ifd_norm.min()) / (ifd_norm.max() - ifd_norm.min() + 1e-10)
        
        # Show dim ifdgram as background
        ax.imshow(ifd_norm, cmap='gray', vmin=0, vmax=4, aspect='auto', origin='upper', extent=[0, w, 0, h])
        
        # Draw contours at t_disp = 0 (red)
        ax.contour(t_d, levels=[0], colors=['red'], linewidths=0.5, origin='upper', extent=[0, w, 0, h])
        
        # Draw contours at f_disp = 0 (blue)  
        ax.contour(f_d, levels=[0], colors=['blue'], linewidths=0.5, origin='upper', extent=[0, w, 0, h])
        
        # Sign change detection: find 2x2 blocks where both t_d and f_d cross zero
        # Check diagonal sign changes in 2x2 blocks
        t_sign_change = ((t_d[:-1,:-1] * t_d[1:,1:] < 0) | (t_d[:-1,1:] * t_d[1:,:-1] < 0))
        f_sign_change = ((f_d[:-1,:-1] * f_d[1:,1:] < 0) | (f_d[:-1,1:] * f_d[1:,:-1] < 0))
        zeros_mask = t_sign_change & f_sign_change
        
        # Compute Jacobian determinant to filter true singularities
        # det(J) = ∂t_d/∂x * ∂f_d/∂y - ∂t_d/∂y * ∂f_d/∂x
        dt_dx = np.diff(t_d, axis=1)  # ∂t_d/∂x
        dt_dy = np.diff(t_d, axis=0)  # ∂t_d/∂y
        df_dx = np.diff(f_d, axis=1)  # ∂f_d/∂x
        df_dy = np.diff(f_d, axis=0)  # ∂f_d/∂y
        
        # Average derivatives at 2x2 block centers
        dt_dx_avg = (dt_dx[:-1,:] + dt_dx[1:,:]) / 2
        dt_dy_avg = (dt_dy[:,:-1] + dt_dy[:,1:]) / 2
        df_dx_avg = (df_dx[:-1,:] + df_dx[1:,:]) / 2
        df_dy_avg = (df_dy[:,:-1] + df_dy[:,1:]) / 2
        
        # Jacobian determinant
        det_J = dt_dx_avg * df_dy_avg - dt_dy_avg * df_dx_avg
        
        # True singularities have |det_J| > threshold
        # Threshold based on data statistics
        det_thresh = np.std(np.abs(det_J)) * 0.5
        true_singularities = zeros_mask & (np.abs(det_J) > det_thresh)
        
        # Get coordinates of zeros (offset by 0.5 to center in 2x2 block)
        zero_y, zero_x = np.where(true_singularities)
        zero_x = zero_x + 0.5
        zero_y = zero_y + 0.5
        
        # Get charges (sign of det_J) for coloring
        charges = det_J[true_singularities]
        
        # Plot zeros: positive charge (white), negative charge (yellow)
        pos_mask = charges > 0
        neg_mask = charges < 0
        if np.any(pos_mask):
            ax.scatter(zero_x[pos_mask], h - zero_y[pos_mask], s=15, c='white', marker='o', edgecolors='black', linewidths=0.5, zorder=10)
        if np.any(neg_mask):
            ax.scatter(zero_x[neg_mask], h - zero_y[neg_mask], s=15, c='yellow', marker='o', edgecolors='black', linewidths=0.5, zorder=10)
        
        # Render to array
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgb_array = np.asarray(buf)[:, :, :3]  # Drop alpha
        
        rgb_uint8 = rgb_array.astype(np.uint8)
    
    else:
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
    # Search for both .wav and .WAV extensions
    wav_files = glob.glob(os.path.join(script_dir, "*.wav")) + glob.glob(os.path.join(script_dir, "*.WAV"))
    wav_names = [os.path.basename(f) for f in sorted(set(wav_files))]
    print(f"Looking for WAV files in: {script_dir}")
    print(f"Found: {wav_names}")
    return jsonify(wav_names)


@app.route('/api/compute', methods=['POST'])
def compute():
    """Compute spectrogram with given parameters."""
    import sys
    data = request.get_json()
    print(f"=== API COMPUTE CALLED ===", file=sys.stderr, flush=True)
    
    wav_name = data.get('wav')
    sigmas = data.get('sigmas', [1.0, 2.0, 3.0])
    compute_type = data.get('type', 'ifdgram')  # 'ifdgram' or 'sonogram'
    print(f"Computing: {wav_name} type={compute_type}", file=sys.stderr, flush=True)
    fft_size = data.get('fft_size', 1024)
    step_size = data.get('step_size', 72)
    superres = data.get('superres', 1)
    lock_t = data.get('lock_t', 5)
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
        if step_size < 1 or step_size > fft_size:
            return jsonify({'error': f'Step size must be between 1 and {fft_size}'}), 400
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_files = glob.glob(os.path.join(script_dir, "*.wav")) + glob.glob(os.path.join(script_dir, "*.WAV"))
    wav_names = [os.path.basename(f) for f in sorted(set(wav_files))]
    
    print("Starting spectrogram server at http://localhost:8000")
    print(f"Working directory: {script_dir}")
    print(f"WAV files found: {wav_names if wav_names else 'NONE - add .wav files to this folder'}")
    print("Open http://localhost:8000 in your browser")
    app.run(host='127.0.0.1', port=8000, debug=False, threaded=True)
