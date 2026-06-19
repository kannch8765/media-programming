# mp_ex2.ipynb — Exercise Notebook (Power Spectrum and Spectrogram)

This is the **assignment** notebook for the week, not a worked example. It contains two empty code cells where the student must write code to produce two specific figures, save them, then submit the notebook with the figures rendered.

The instructions are in Japanese; below is a plain-English summary plus an explanation of the boilerplate code already present in each cell.

## Setup

```python
urllib.request.urlretrieve(
    'https://.../ex-Violin.wav',
    './sound/ex-Violin.wav'
)
urllib.request.urlretrieve(
    'https://.../ex-Violin-music.wav',
    './sound/ex-Violin-music.wav'
)
```

Downloads two violin samples into `sound/`. Re-run on each fresh Colab session.

## Task 1 — Power spectrum of a single note

> Open `sound/ex-Violin.wav`, compute its power spectrum, plot amplitude (dB) vs. frequency (Hz), and save the figure as `ex2_1.png` in the notebook's directory. Use `wave.getframerate()` to get the sampling rate. Use `p_0 = 1` for the dB calculation. Add axis ticks and labels.

The starter code already loads the file:

```python
%matplotlib inline
import wave
import numpy as np
import matplotlib.pyplot as plt

wavfile = wave.open('sound/ex-Violin.wav', 'rb')
x = wavfile.readframes(wavfile.getnframes())
sampling_rate = wavfile.getframerate()
sample_size = wavfile.getsampwidth()
wavfile.close()
```

What each line does:
- `%matplotlib inline` makes plots render inside the notebook instead of in a separate window.
- `wave.open(path, 'rb')` opens the WAV file in read mode.
- `readframes(n)` reads `n` frames of raw audio bytes. Passing `getnframes()` reads the whole file.
- `getframerate()` returns the sampling rate in Hz.
- `getsampwidth()` returns bytes per sample — useful for normalising the data.
- `wavfile.close()` releases the file handle.

What you need to add (drawing on notebooks 1 and 3):
1. Convert `x` from bytes to a normalised NumPy array (`np.frombuffer`, divide by `2**(8*sample_size)/2`).
2. Compute the FFT with `np.fft.fft`.
3. Compute the amplitude in dB: `20 * np.log10(np.abs(X) + 1e-10)`.
4. Build a frequency axis with `np.fft.fftfreq` (or by hand).
5. Keep only the lower half of the spectrum (below Nyquist).
6. Plot with `plt.plot`, set xlabel/ylabel/grid, then `plt.savefig('ex2_1.png')`.

## Task 2 — Spectrogram of violin music

> Open `sound/Violin-music.wav`, draw its spectrogram, and save the figure as `ex2_2.png`. Parameters:
> - Window size: 512 samples
> - Overlap: 32 frames
> - Window function: Hanning

The starter code is again ready:

```python
%matplotlib inline
import wave
import numpy as np
import matplotlib.pyplot as plt

wavfile = wave.open("sound/ex-Violin-music.wav", "rb")
x = wavfile.readframes(wavfile.getnframes())
x = np.frombuffer(x, dtype="int16")
sampling_rate = wavfile.getframerate()
wavfile.close()
```

Same as task 1, except the data is already converted to `int16`.

What you need to add (drawing on notebook 4):
1. Build a Hanning window: `np.hanning(512)` — note that the notebook instructions ask for **Hanning**, not Hamming like in notebook 4. They are different windows; pick the right one.
2. Call `plt.specgram(x, NFFT=512, Fs=sampling_rate, noverlap=32, window=hanningWindow, cmap='turbo')`.
3. Label axes (`time [sec]`, `frequency [Hz]`) and save as `ex2_2.png`.

> Tip: `noverlap=32` is **much smaller** than the `N//2 = 256` used in notebook 4. That means the time slices move forward by `N - noverlap = 480` samples at a time, giving a coarser but faster spectrogram.

## Playback cells

The notebook ends with two `IPython.display.Audio(...)` cells so you can listen to the samples — useful for sanity-checking whether the spectrum or spectrogram you produced looks consistent with what you hear.

## Key takeaways

- This notebook is a **scaffold**: every line you need is already imported; you only add the analysis and plotting code.
- The two tasks directly reuse the FFT (notebook 3) and STFT (notebook 4) techniques.
- Window size, overlap, and window function are the key knobs that change the look of a spectrogram.