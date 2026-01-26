# Interactive Reassigned Spectrogram Viewer

A Python-based tool for computing and visualizing reassigned spectrograms with an interactive WebGL viewer. Based on the algorithm described in Gardner & Magnasco, PNAS 2006.

![Spectrogram Viewer](screenshot.png)
*Interactive spectrogram viewer with reassigned spectrogram (ifdgram) display.*

## What is a Reassigned Spectrogram?

Standard spectrograms suffer from a tradeoff between time and frequency resolution. Reassigned spectrograms (also called "remapped sonograms") use the phase information to sharpen the representation, producing cleaner spectral lines without sacrificing resolution.
This visualizer computes the spectrogram at multiple timescales (sigma values) and combines them. This implementation displays three timescales as RGB channels, allowing you to see which features are consistent across scales.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Files

- **ifdv.py** - Core algorithm for computing reassigned spectrograms
- **server.py** - Flask server for interactive viewing with on-demand computation
- **viewer.html** - WebGL-based viewer with max-pooling for large images
- **requirements.txt** - Python dependencies

## Usage

### Interactive Viewer (Recommended)

Start the Flask server:
```bash
source venv/bin/activate
python server.py
```

Open http://localhost:8000 in your browser.

**Controls:**

*Compute Parameters (require recomputation via Compute button):*
- **WAV selector** - Choose which audio file to analyze (parameters reset to defaults when switching)
- **ifdgram/sonogram** - Toggle between reassigned and standard spectrogram
- **σ (ms)** - Three sigma values for R, G, B channels
- **FFT** - FFT window size (256, 512, 1024, 2048, 4096)
- **Step** - Samples between windows (smaller = higher time resolution)
- **Superres** - Frequency superresolution factor (1-10)
- **TL** - Temporal locking window (stray point removal in time)
- **FL** - Frequency locking window (stray point removal in frequency)
- **LogOff** - Log scale offset for `log(amp + offset)` contrast

*Display Parameters (real-time WebGL, no recomputation needed):*
- **CropF** - Frequency crop factor (0.1-1.0, show low frequencies)
- **Min/Max** - Intensity thresholds for display contrast
- **X/Y zoom** - Independent time and frequency zoom

*Navigation:*
- **Mouse drag** - Pan along time axis
- **Mouse wheel** - Zoom time axis at cursor position
- **W/S keys** - Frequency (Y) zoom in/out
- **A/D keys** - Time (X) zoom in/out
- **R key** - Reset view to fit image

*Audio Playback:*
- **Play** - Play the audio for the currently visible time range
- **Stop** - Stop audio playback

**Caching:** Computed spectrograms are cached on the server. Switching between files or toggling ifdgram/sonogram with the same parameters is instant after the first computation.

## Parameters

### Sigma (σ)
The temporal resolution in milliseconds. Controls the time-frequency tradeoff:
- **Small σ (0.5-1ms)** - Better time resolution, sees clicks and transients
- **Large σ (3-5ms)** - Better frequency resolution, sees tonal content
- **Intermediate** - Usually best for most signals

The RGB display shows three sigmas simultaneously:
- **Red** - First sigma value (typically smallest)
- **Green** - Second sigma value
- **Blue** - Third sigma value (typically largest)

White/gray regions indicate features present at all timescales.

### FFT Size
The FFT window size controls the frequency resolution:
- **Smaller (256, 512)** - Better time resolution, coarser frequency bins
- **Larger (2048, 4096)** - Better frequency resolution, smoother in time

### Step Size
The number of samples between successive windows (default: 72). `overlap = fft_size - step_size`.
- **Smaller step** - More overlap, sharper lines, but wider images and slower computation
- **Larger step** - Less overlap, faster computation, narrower images

### Superresolution
Frequency superresolution factor (1-10). Increases the number of frequency bins in the output by interpolating during the reassignment process. Higher values show more frequency detail but increase computation time.

### Locking Windows (TL, FL)
Control the removal of "stray points" in the reassigned spectrogram:
- **TL (Temporal Locking)** - Maximum allowed displacement in time (default: 5)
- **FL (Frequency Locking)** - Maximum allowed displacement in frequency (default: 5)
- **Smaller values** - Sharper lines but may introduce gaps
- **Larger values** - Preserves more signal but may be noisier

### CropF (Frequency Crop)
Controls what fraction of the frequency spectrum to display (0.1-1.0):
- **0.4** (default) - Show bottom 40% of frequency range
- **1.0** - Show full frequency spectrum
- **0.5** - Show bottom half (lower frequencies)

This is a real-time WebGL control - adjust instantly without recomputing.

### Min/Max (Intensity Thresholds)
Real-time display controls for adjusting contrast:
- **Min** (default 0.05) - Values below this threshold become black
- **Max** (default 0.80) - Values above this threshold become maximum intensity

Colors are rescaled between these thresholds, allowing interactive adjustment of dynamic range without recomputing the spectrogram.

### LogOff (Log Offset)
Controls the offset in the log scale transformation `log(amp + offset)`:
- **Lower values (0.01-0.1)** - More contrast for weak signals
- **Higher values (0.5-1.0)** - Compressed dynamic range
- Default: 0.3

## WebGL Max-Pooling

The viewer uses a WebGL shader that performs max-pooling when zoomed out. This means when multiple spectrogram pixels are combined into one screen pixel, the brightest value is shown rather than the average. This preserves signal peaks that might otherwise be diluted.

## Limitations

- WebGL has a maximum texture size (typically 16384 pixels). Very long recordings are automatically truncated to fit within this limit based on the step size.
- The interactive viewer requires a local server due to browser security restrictions on loading local files into WebGL.

## References

- Gardner, T. J., & Magnasco, M. O. (2006). Efficient and robust spectro-temporal analysis using sparse signal reconstruction. *PNAS*, 103(16), 6094-6099.
- Gardner, T. J., & Magnasco, M. O. (2005). Sparse time-frequency representations. *J. Acoust. Soc. Am.*, 118(3), 2095-2104.
