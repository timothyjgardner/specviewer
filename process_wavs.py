"""
Batch process WAV files to generate multi-timescale spectrograms.

Loads all WAV files in the current directory, computes ifdv at three
timescales (0.5ms, 1ms, 2ms) for each, and saves RGB images where
R=0.5ms, G=1ms, B=2ms. Generates both sonogram and ifdgram images.
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy.io import wavfile

from ifdv import ifdv


def process_wav_file(wav_path, output_dir=None):
    """
    Process a single WAV file and save ifdgram and sonogram images.
    
    Parameters
    ----------
    wav_path : str
        Path to the WAV file.
    output_dir : str, optional
        Directory to save output images. If None, saves in same directory as WAV.
    
    Returns
    -------
    ifdgram_path : str
        Path to the saved ifdgram image.
    sonogram_path : str
        Path to the saved sonogram image.
    """
    # Load WAV file
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
    
    # Compute ifdgram with typical parameters
    n = 1024
    overlap = n - 256  # High overlap for sharper lines
    zoom_t = 1
    zoom_f = 1
    tl = 15
    fl = 15
    
    # WebGL max texture size limit
    MAX_IMAGE_WIDTH = 16384
    step = n - overlap
    max_samples = (MAX_IMAGE_WIDTH - 1) * step + n
    
    # If recording would exceed max width, use only first half
    original_duration = len(data) / sampling_rate
    if len(data) > max_samples:
        data = data[:len(data) // 2]
        print(f"Processing: {os.path.basename(wav_path)}")
        print(f"  Original duration: {original_duration:.2f}s - TRUNCATED to first half")
    else:
        print(f"Processing: {os.path.basename(wav_path)}")
    
    # Three timescales for RGB channels
    sigmas = [1, 2.0, 3.0]  # R=0.5ms, G=1ms, B=2ms
    
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Duration: {len(data) / sampling_rate:.2f} seconds")
    print(f"  Samples: {len(data)}")
    
    # Compute ifdgram and sonogram for each timescale
    ifdgrams = []
    sonograms = []
    for sigma in sigmas:
        print(f"  Computing sigma={sigma}ms...")
        ifdgram, sonogram, dx = ifdv(data, sampling_rate, n, overlap, sigma, zoom_t, zoom_f, tl, fl)
        ifdgrams.append(ifdgram)
        sonograms.append(sonogram)
    
    # Create output paths
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)
    ifdgram_path = os.path.join(output_dir, f"{base_name}_ifdgram.png")
    sonogram_path = os.path.join(output_dir, f"{base_name}_sonogram.png")
    
    # Only show bottom half of frequency range (lower frequencies)
    half_freq = ifdgrams[0].shape[0] // 2
    ifdgrams = [img[half_freq:] for img in ifdgrams]
    sonograms = [np.abs(img[half_freq:]) for img in sonograms]
    
    # Normalize each channel to 0-1 range
    def normalize(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return np.zeros_like(img)
    
    # Process ifdgrams: apply log scaling and normalize
    ifdgram_channels = [normalize(np.log(img + .1)) for img in ifdgrams]
    ifdgram_rgb = np.stack(ifdgram_channels, axis=-1)
    
    # Process sonograms: apply log scaling and normalize
    sonogram_channels = [normalize(np.log(img + .1)) for img in sonograms]
    sonogram_rgb = np.stack(sonogram_channels, axis=-1)
    
    # Convert to 8-bit and save directly with PIL (no matplotlib rescaling)
    ifdgram_uint8 = (ifdgram_rgb * 255).astype(np.uint8)
    sonogram_uint8 = (sonogram_rgb * 255).astype(np.uint8)
    
    # Save ifdgram
    Image.fromarray(ifdgram_uint8).save(ifdgram_path)
    print(f"  Saved: {ifdgram_path}")
    
    # Save sonogram
    Image.fromarray(sonogram_uint8).save(sonogram_path)
    print(f"  Saved: {sonogram_path}")
    
    return ifdgram_path, sonogram_path


def main():
    """Process all WAV files in the current directory."""
    # Find all WAV files in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_files = glob.glob(os.path.join(script_dir, "*.wav"))
    
    if not wav_files:
        print("No WAV files found in the current directory.")
        return
    
    print(f"Found {len(wav_files)} WAV file(s) to process:\n")
    
    processed = []
    for wav_path in sorted(wav_files):
        try:
            ifdgram_path, sonogram_path = process_wav_file(wav_path)
            processed.append((ifdgram_path, sonogram_path))
            print()
        except Exception as e:
            print(f"  Error processing {wav_path}: {e}\n")
    
    print(f"Processing complete. Generated {len(processed) * 2} image(s) from {len(processed)} WAV file(s).")


if __name__ == "__main__":
    main()
