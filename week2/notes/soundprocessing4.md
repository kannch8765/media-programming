# SoundProcessing4.ipynb — Short-Time Fourier Transform and Spectrograms

Notebook 3 applied the FFT to a single short sound ("あ"). But what if the audio contains frequencies that change over time — like a spoken word or music? Notebook 4 introduces the **Short-Time Fourier Transform (STFT)**, which chops the waveform into small overlapping windows and runs an FFT on each. Stacking the results column-by-column produces a **spectrogram**.

Sample files:
- `4-female_aiueo.wav` — woman saying "あいうえお"
- `4-male_aiueo.wav` — man saying "あいうえお"

## Setup

```python
urllib.request.urlretrieve(...female_aiueo..., './sound/4-female_aiueo.wav')
urllib.request.urlretrieve(...male_aiueo...,   './sound/4-male_aiueo.wav')
```

## Reading the file

```python
wavfile = wave.open("sound/4-female_aiueo.wav", "rb")
x = wavfile.readframes(wavfile.getnframes())
x = np.frombuffer(x, dtype="int16")
sampling_rate = wavfile.getframerate()
wavfile.close()
```

Same WAV-reading pipeline used in earlier notebooks — bytes → `int16` array.

## 1. STFT overview

The notebook outlines the four-step recipe:

1. **Slice** the waveform into short overlapping frames.
2. **Window** each frame with a function whose weight is largest at the centre and tapers to zero at the edges.
3. Run the **FFT** on each windowed frame.
4. Stack the resulting log-power spectra side-by-side, with time on the x-axis and frequency on the y-axis.

This produces a spectrogram, where brighter colours (in `turbo`/`viridis`/`magma` colormaps) mean stronger frequency components at that time.

## 2. Computing the spectrogram with `matplotlib.pyplot.specgram`

```python
N = 512
hammingWindow = np.hamming(N)

pxx, freqs, bins, im = plt.specgram(
    x,
    NFFT=N,
    Fs=sampling_rate,
    noverlap=N//2,
    window=hammingWindow,
    cmap='turbo'
)
```

- `N = 512` — the window length in samples. With a 16 kHz file this is 32 ms of audio per frame, a typical compromise between time and frequency resolution.
- `np.hamming(N)` — the **Hamming window** (length `N`). It multiplies each frame element-wise, suppressing the discontinuities at the slice boundaries and reducing spectral leakage.
- `plt.specgram` does the four STFT steps automatically:
  - `NFFT=N` — FFT size = window size.
  - `noverlap=N//2` — successive frames overlap by 256 samples, giving smoother transitions.
  - `window=hammingWindow` — uses our explicit Hamming window.
  - `cmap='turbo'` — colormap for the heatmap (red = high power, blue = low power).
- It returns:
  - `pxx` — 2-D array of power values `[freq × time]`.
  - `freqs` — y-axis (frequency) labels.
  - `bins` — x-axis (time) labels.
  - `im` — the `Image` artist, useful for adding a colour bar.

```python
plt.xlabel("time [second]")
plt.ylabel("frequency [Hz]")
plt.savefig('fig/SoundProcessing4-1.png')
```

Labels the axes and saves the figure.

## 3. STFT at one specific time slice

To show how the spectrogram is built from one frame, the notebook isolates `t = 1.0 s`:

```python
ShortWav = x[16000:16000+N]   # 16 kHz × 1 s = 16000 samples
hammingWindow = np.hamming(N)
WeightedShortWav = ShortWav * hammingWindow
PowerSpectol = 20*np.log10(np.abs(np.fft.fft(WeightedShortWav)) + 1e-10)
x_axis = np.fft.fftfreq(N, d=1.0/sampling_rate)
```

- `ShortWav` — the 512-sample slice starting at frame 16 000.
- `WeightedShortWav` — element-wise product with the Hamming window. The window zeroes (or near-zeros) the values at the edges of the slice, so each frame reflects only what was happening near that moment.
- `np.fft.fft(...)` — frequency decomposition of the windowed slice.
- `np.abs(...) + 1e-10` and `20·log10(...)` — magnitude → power spectrum in dB. The `1e-10` is the same `log(0)` guard as before.
- `np.fft.fftfreq(N, d=1/Fs)` — generates the matching frequency axis: `[-Fs/2, …, -1, 0, 1, …, Fs/2]`. Taking only the first half (`x_axis[:N/2]`) gives the non-negative frequencies.

A 2×2 grid of plots summarises the four steps:

| Subplot | Title | What it shows |
| --- | --- | --- |
| (1) | `a) Sliced wave` | The raw 512-sample slice from the waveform. |
| (2) | `b) Window function` | The Hamming window itself (a smooth bell shape). |
| (3) | `c) Matrix multiplication result of a) and b)` | The slice * window — the actual signal fed into the FFT. |
| (4) | `d) Power Spectol of c)` | The log-power spectrum of the windowed slice. Peaks correspond to harmonics of the vowel being spoken. |

Every vertical strip in the final spectrogram is essentially a copy of plot (d), shifted rightward in time.

## Key takeaways

- STFT answers "what frequencies are present at each moment in time?" — plain FFT can only answer "what frequencies are present overall?"
- A **window function** (Hamming, Hanning, etc.) is multiplied with each frame to suppress edge artefacts.
- `plt.specgram` is the convenient one-liner; manual control comes from slicing + `np.fft.fft` per window.
- Time-frequency resolution is a trade-off: longer windows → better frequency resolution but worse time resolution; shorter windows → the opposite.