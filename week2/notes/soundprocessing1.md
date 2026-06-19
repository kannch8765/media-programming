# SoundProcessing1.ipynb — Sound Representation (Read / Play / Visualize)

This notebook is the introductory one for the "Sound Information Processing" series. The sample audio is a violin solo (`sound/1-Violin-music.wav`). The notebook walks through reading a WAV file, checking its parameters, plotting the waveform, and zooming into a 1-millisecond slice to reveal that the waveform is actually a sequence of discrete samples.

## Setup cell — download the audio

```python
import urllib.request
!mkdir -p fig sound
urllib.request.urlretrieve('https://.../1-Violin-music.wav', './sound/1-Violin-music.wav')
```

- `urllib.request.urlretrieve(url, path)` downloads a remote file directly to disk.
- `!mkdir -p fig sound` is a shell command that creates two folders (`fig` for output images, `sound` for audio files). `-p` makes the command succeed even if the folders already exist.

## Audio playback

```python
import IPython
IPython.display.Audio('sound/1-Violin-music.wav')
```

- This wraps the local WAV file in an HTML5 `<audio>` player so it can be played inside a Jupyter notebook.

## Reading the file

```python
import wave
wavfile = wave.open('sound/1-Violin-music.wav', 'rb')
```

- `wave.open(path, 'rb')` opens a WAV file in binary-read mode. The returned `wave.Wave_read` object lets you read the raw bytes plus the file's metadata.

## Inspecting the file's parameters

```python
print("Channel num : ", wavfile.getnchannels())
print("Sample size : ", wavfile.getsampwidth())
print("Sampling rate : ", wavfile.getframerate())
print("Frame num : ", wavfile.getnframes())
print("Sec : ", float(wavfile.getnframes()) / wavfile.getframerate())
print("Prams : ", wavfile.getparams())
```

- `getnchannels()` — 1 for mono, 2 for stereo.
- `getsampwidth()` — bytes per sample. `2` means 16-bit (since 1 byte = 8 bits).
- `getframerate()` — number of samples per second (Hz).
- `getnframes()` — total sample count.
- `nframes / framerate` — total duration in seconds.
- `getparams()` — prints everything above in one tuple.

For this violin file the result is mono, 16-bit, 48 kHz, 536 465 frames ≈ 11.18 s.

## Waveform plotting (Section 2.1)

```python
data = wavfile.readframes(wavfile.getnframes())
sampling_rate = wavfile.getframerate()
sample_size = wavfile.getsampwidth()
wavfile.close()
```

- `readframes(n)` reads `n` frames as raw bytes. Passing `getnframes()` reads the entire file.
- After capturing `sampling_rate` and `sample_size`, the file is closed.

```python
data = np.frombuffer(data, dtype="int16")
amp = 2**(8 * sample_size) / 2
data = data / amp
```

- `np.frombuffer(..., dtype="int16")` interprets the raw bytes as 16-bit signed integers (one per sample, in the range −32 768 … 32 767).
- `amp` is half the maximum representable value — i.e. the absolute value of the largest signed integer we can store. Dividing by it normalises the waveform to the range `[-1.0, 1.0]`.

```python
x = np.linspace(0, len(data)/sampling_rate, len(data))
```

- Builds the time-axis: starts at 0 s, ends at `len(data)/sampling_rate` s (the duration), with one tick per sample. Each step is exactly `1/sampling_rate` seconds apart.

```python
fig = plt.figure(figsize=(7, 3))
ax = plt.subplot()
ax.plot(x, data)
ax.set_xlabel("time [sec.]")
ax.set_ylabel("amplitude")
ax.grid()
fig.tight_layout()
plt.savefig('fig/SoundProcessing1-1.png')
```

- `figsize=(7, 3)` sets the canvas size in inches.
- `plt.subplot()` claims a single axes on the figure.
- `ax.plot(x, data)` draws amplitude vs. time.
- `tight_layout()` trims padding so labels are not clipped.
- `plt.savefig(...)` writes the figure to disk as a PNG.

## Zooming in 1 ms (Section 2.2)

```python
start = 1.0
duration = 0.001
ax.plot(x[round(start*sampling_rate):round(start*sampling_rate + duration*sampling_rate)],
        data[round(start*sampling_rate):round(start*sampling_rate + duration*sampling_rate)],
        marker="o")
```

- `start*sampling_rate` converts the start time (1.0 s) into a sample index.
- The slice `[start : start + duration*sampling_rate]` extracts roughly 48 samples (since 48 kHz × 0.001 s ≈ 48), corresponding to a 1-millisecond window of audio.
- Adding `marker="o"` makes each sample a visible dot, so it is obvious that the waveform is not continuous but a series of discrete measurements.
- `plt.setp(ax.get_xticklabels(), rotation=30, ...)` rotates the time labels 30° so they do not overlap.

## Key takeaways

- A WAV file stores: sample rate, bit depth, channel count, and the raw sample values.
- The waveform is discrete — every value comes from one sample, even if it looks continuous at a small scale.
- To plot it against time, multiply each sample index by `1 / sampling_rate`.
- Normalising to `[-1, 1]` by dividing by `2^(8*sample_size)/2` is the standard way to express amplitude regardless of bit depth.