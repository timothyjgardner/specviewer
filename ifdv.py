"""
Remapped sonograms (reassigned spectrograms)

Based on Gardner & Magnasco PNAS 2006
Converted from MATLAB to Python

For lack of a better name, "ifdgram" is the new sonogram.
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram


def ifdv(s, sampling, n, overlap, sigma, zoom_t, zoom_f, tl, fl):
    """
    Compute reassigned spectrogram.
    
    Parameters
    ----------
    s : array_like
        Signal to be analyzed.
    sampling : float
        Sampling rate of the signal (Hz).
    n : int
        Number of filters in the filter-bank (window size).
    overlap : int
        Number of samples to overlap in each successive window.
        The number of time points in the final image is proportional to
        len(s) / (n - overlap). Higher overlap results in sharper lines.
    sigma : float
        Temporal resolution of the analysis in milliseconds.
        n should be larger than 5 * sampling * (sigma / 1000).
        Choose sigma small to represent sound in a time-like fashion (as a
        series of clicks), or sigma large to represent sound in a frequency-like
        fashion (as a series of tones). For most signals, intermediate values
        are best.
    zoom_t : float
        Temporal resolution of the final image (typically 1).
    zoom_f : float
        Frequency resolution of final image. Resolution = zoom_f * (n / 2).
        It is typically useful to set zoom_f greater than one.
    tl : float
        Temporal locking window in pixels.
    fl : float
        Frequency locking window in pixels.
        
        When the remapping moves a pixel by more than TL or FL, that pixel
        acquires zero weight. For discussion of the locking window, see
        Gardner & Magnasco, J. Acoust. Soc. Am. 2005.
        
        When these parameters are small (order 1), "stray" points are removed
        and the lines are sharpened. If these parameters are too small, lines
        become too thin and appear discontinuous.
    
    Returns
    -------
    ifdgram : ndarray
        The reassigned spectrogram (remapped sonogram).
    sonogram : ndarray
        The standard spectrogram.
    dx : ndarray
        Displacement factors according to the remapping algorithm.
    
    Notes
    -----
    Typical parameters: sampling=44100, n=1024, overlap=1010,
    sigma=2, zoom_t=1, zoom_f=3, fl=5, tl=5
    
    Implementation note:
    The best results will come from calculating an ifdgram for many values of
    sigma (0.5:0.1:3.5 for example), then combining by multiplying together
    images with neighboring values of sigma, and adding them all together.
    Rationale for this is given in Gardner & Magnasco PNAS 2006.
    
    Example
    -------
    >>> # Compute an ifdgram of 100ms of white noise
    >>> s = np.random.rand(4000) - 0.5
    >>> ifdgram, sonogram, dx = ifdv(s, 44100, 1024, 1020, 2, 1, 1, 2, 2)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(np.log(ifdgram + 3), cmap='hot', aspect='auto')
    >>> plt.colorbar()
    >>> plt.show()
    
    Comment: Log scaling of the ifdgram may be optimal for most sounds.
    """
    s = np.asarray(s).flatten()
    
    factor = float(n - overlap)
    t = np.arange(-n // 2, n // 2)  # Centered window
    
    # Gaussian and first derivative as windows
    sigma_samples = (sigma / 1000) * sampling
    w = np.exp(-(t / sigma_samples) ** 2)
    dw = w * (t / (sigma_samples ** 2)) * -2
    
    # Compute spectrograms using custom spectrogram function
    # that mimics MATLAB's specgram behavior
    q = _specgram(s, n, w, overlap) + np.finfo(float).eps
    q2 = _specgram(s, n, dw, overlap) + np.finfo(float).eps
    
    f_size, t_size = q.shape
    
    # Displacement according to the remapping algorithm
    dx = (q2 / q) / (2 * np.pi)
    
    sonogram = q.copy()
    
    # Create frequency and time index arrays
    to = np.tile(np.arange(1, t_size + 1), (f_size, 1))
    fo = np.tile((0.5 * np.arange(1, f_size + 1) - 1).reshape(-1, 1) / (f_size - 1), (1, t_size))
    
    # Calculate frequency and temporal displacement factors
    f_est = ((fo - np.imag(dx)) * n) + 1
    t_est = to - (np.pi * sigma_samples * sigma_samples) * np.real(dx) / factor
    
    # Reference grids for locking window
    tref = zoom_t * np.tile(np.arange(1, t_size + 1), (f_size, 1))
    fref = zoom_f * np.tile(np.arange(1, f_size + 1).reshape(-1, 1), (1, t_size))
    
    # Rescale dimensions
    f_final = int(f_size * zoom_f)
    t_final = int(t_size * zoom_t)
    
    # Round displacement estimates
    f_e = np.round(zoom_f * f_est).astype(int)
    t_e = np.round(zoom_t * t_est).astype(int)
    
    # Create a copy of q for modification
    q_mod = np.abs(q.copy())
    
    # Set to zero points mapped out of the image
    q_mod[(f_e < 1) | (f_e > f_final)] = 0
    q_mod[(t_e < 1) | (t_e > t_final)] = 0
    
    # Clip indices to valid range
    f_e = np.clip(f_e, 1, f_final)
    t_e = np.clip(t_e, 1, t_final)
    
    # Remove "stray points" using locking window
    q_mod[np.abs(f_e - fref) > fl] = 0
    q_mod[np.abs(t_e - tref) > tl] = 0
    
    # Convert to 0-based indexing for Python
    f_e_flat = (f_e - 1).flatten()
    t_e_flat = (t_e - 1).flatten()
    q_flat = q_mod.flatten()
    
    # Build the fully remapped sonogram using accumarray equivalent
    # np.add.at allows accumulation at repeated indices
    newq = np.zeros((f_final, t_final))
    np.add.at(newq, (f_e_flat, t_e_flat), q_flat)
    
    # Build time-only reassigned spectrogram (reassign in time, keep original frequency)
    f_orig = np.clip(fref.astype(int), 1, f_final)  # Original frequency indices
    f_orig_flat = (f_orig - 1).flatten()
    newq_t = np.zeros((f_final, t_final))
    np.add.at(newq_t, (f_orig_flat, t_e_flat), q_flat)
    
    # Build frequency-only reassigned spectrogram (reassign in frequency, keep original time)
    t_orig = np.clip(tref.astype(int), 1, t_final)  # Original time indices
    t_orig_flat = (t_orig - 1).flatten()
    newq_f = np.zeros((f_final, t_final))
    np.add.at(newq_f, (f_e_flat, t_orig_flat), q_flat)
    
    # Flip so low frequency is at the bottom
    ifdgram = np.flipud(newq)
    ifdgram_t = np.flipud(newq_t)  # Time-only reassignment
    ifdgram_f = np.flipud(newq_f)  # Frequency-only reassignment
    sonogram = np.flipud(sonogram)
    
    # Compute displacement fields (how far each point moves)
    # Time displacement: difference between reassigned time and original time
    t_displacement = t_est - to
    # Frequency displacement: difference between reassigned freq and original freq  
    f_displacement = f_est - (fo * n + 1)
    
    # Flip to match spectrogram orientation
    t_displacement = np.flipud(t_displacement)
    f_displacement = np.flipud(f_displacement)
    
    return ifdgram, sonogram, dx, t_displacement, f_displacement, ifdgram_t, ifdgram_f


def _specgram(x, nfft, window, noverlap):
    """
    Compute spectrogram similar to MATLAB's specgram function.
    
    Parameters
    ----------
    x : array_like
        Input signal.
    nfft : int
        FFT length and window size.
    window : array_like
        Window function.
    noverlap : int
        Number of overlapping samples.
    
    Returns
    -------
    S : ndarray
        Complex spectrogram (positive frequencies only).
    """
    x = np.asarray(x).flatten()
    window = np.asarray(window).flatten()
    
    step = nfft - noverlap
    
    # Pad signal if necessary
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)), mode='constant')
    
    # Number of segments
    num_segments = max(1, (len(x) - nfft) // step + 1)
    
    # Initialize output (positive frequencies only: nfft//2 + 1, but we use nfft//2 to match MATLAB)
    num_freqs = nfft // 2
    S = np.zeros((num_freqs, num_segments), dtype=complex)
    
    for i in range(num_segments):
        start = i * step
        segment = x[start:start + nfft]
        if len(segment) < nfft:
            segment = np.pad(segment, (0, nfft - len(segment)), mode='constant')
        
        # Apply window and compute FFT
        windowed = segment * window
        fft_result = np.fft.fft(windowed, nfft)
        
        # Keep only positive frequencies (excluding DC and Nyquist for consistency)
        S[:, i] = fft_result[1:num_freqs + 1]
    
    return S


if __name__ == "__main__":
    # Example: Compute an ifdgram of 100ms of white noise
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    s = np.random.rand(4000) - 0.5
    
    ifdgram, sonogram, dx = ifdv(s, 44100, 1024, 1020, 2, 1, 1, 2, 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(np.log(np.abs(sonogram) + 3), cmap='hot', aspect='auto')
    axes[0].set_title('Standard Sonogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency')
    
    axes[1].imshow(np.log(ifdgram + 3), cmap='hot', aspect='auto')
    axes[1].set_title('Reassigned Spectrogram (ifdgram)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('ifdv_example.png', dpi=150)
    plt.show()
    
    print("Done! Example saved to ifdv_example.png")