# SoundProcessing3.ipynb — Frequency Decomposition, Fourier Transform, Inverse Fourier Transform

This notebook moves from the time domain to the frequency domain. It uses four voice recordings:
- `3-female_a.wav`, `3-female_i.wav` — a woman saying "あ" / "い"
- `3-male_a.wav`, `3-male_i.wav` — a man saying "あ" / "い"

The question it tries to answer: how can we recognise the vowel from a waveform when male and female voices look totally different in the time domain?

## Setup

```python
urllib.request.urlretrieve(...4 wav files...)
```

Downloads the four sample recordings.

## 1.1 Reading a WAV and drawing its waveform

```python
wavfile = wave.open('sound/3-female_a.wav', 'rb')
x = wavfile.readframes(wavfile.getnframes())
sampling_rate = wavfile.getframerate()
sample_size = wavfile.getsampwidth()
wavfile.close()

x = np.frombuffer(x, dtype="int16")
x = x / (2**(8 * sample_size) / 2)
xlabel = np.arange(0, len(x)/sampling_rate, 1.0/sampling_rate)
```

- Standard WAV-reading pipeline (same as notebook 1).
- `np.frombuffer(..., "int16")` decodes the raw bytes as signed 16-bit integers.
- Dividing by `2^(8·sample_size)/2` normalises to `[-1, 1]`.
- `xlabel` is the time axis in seconds, generated with `np.arange`.

```python
ax.plot(xlabel, x)
ax.set_xlabel("time [sec.]")
ax.set_ylabel("amplitude")
ax.set_xlim([0, len(x)/sampling_rate])
ax.grid()
```

Plots amplitude vs. time. Switching the filename in the `wave.open(...)` line lets you compare the four clips.

## 1.2 Discrete Fourier Transform (DFT / FFT)

```python
X = np.fft.fft(x)
print("len(x): ", len(x), " len(X): ", len(X))
```

- `np.fft.fft(x)` computes the Fast Fourier Transform (FFT), the fast algorithm for the Discrete Fourier Transform (DFT).
- The output `X` is a complex-valued array of the same length as `x`. Each entry corresponds to one frequency bin.
- The 10 161-sample female "あ" produces 10 161 complex numbers — confirming that FFT preserves the array size.

## 1.3 Complex numbers encode amplitude and phase

A complex number `c = a + bi` corresponds to a point on the complex plane. When that point rotates around the origin, its imaginary part traces out a sine wave:

- Amplitude: `A = √(a² + b²)`
- Phase: `θ = atan2(b, a)`

So one complex number per frequency bin fully describes the sine wave at that frequency.

## 2.1 Amplitude spectrum (power spectrum)

```python
amplitude = 20*np.log10(np.abs(X) + 1e-10)
```

- `np.abs(X)` extracts the magnitude of each complex value → amplitude spectrum.
- `20 · log10(...)` converts the linear amplitude to decibels (this is the **power spectrum**). The `1e-10` prevents `log10(0)` (which is `-inf`).
- Using dB matches human loudness perception (Weber–Fechner law: sensation is roughly proportional to the log of the stimulus).

```python
ax.plot(amplitude)
ax.set_ylabel("amplitude [dB]")
ax.grid()
```

- The plot shows a symmetric shape around the centre: the left half is the actual spectrum up to the Nyquist frequency (`Fs/2`), and the right half is its mirror image.

```python
ax.plot(amplitude[0:int(len(X)/2)])
```

- Keeps only the first half — the meaningful part below Nyquist.
- The y-axis is still 0 to 5080 indices, but each index maps to a real-world frequency.

```python
x_axis = [i/len(X)*sampling_rate for i in range(0, int(len(X)/2))]
ax.plot(x_axis[0:len(X)//2], amplitude[0:len(X)//2])
ax.set_xlabel("frequency [Hz]")
```

- The frequency axis: `i / N · Fs`. Index 0 → 0 Hz, index `N/2` → `Fs/2`.
- The comment notes that `np.fft.fftfreq(len(X), d=1.0/sampling_rate)` computes the same mapping more concisely.

```python
ax.set_xlim(0, 2000)
ax.set_ylim(-20, 60)
```

- Zooming into the 0–2 000 Hz band reveals a series of sharp peaks at roughly 209 Hz, 420 Hz, 627 Hz, … — integer multiples of 209 Hz.
- These are **harmonics** (倍音) of the fundamental frequency. The lowest is the **fundamental frequency F0**, which defines the perceived pitch.

## 2.2 Phase spectrum

```python
phase = [np.arctan2(int(c.imag), int(c.real)) for c in X[0:int(len(X)/2)]]
ax.plot(x_axis, phase)
ax.set_ylim(-np.pi, np.pi)
ax.set_ylabel("phase spectrum")
```

- `np.arctan2(imag, real)` returns the angle of each complex value in radians (range `[-π, π]`).
- The phase spectrum looks noisy and is hard to interpret visually — but it is essential for reconstruction via the inverse FFT, so we keep it.

## 3. Filtering in the frequency domain

```python
import copy
Th = 1000  # Threshold in Hz

Filtered_X = copy.deepcopy(X)
Filtered_X[int(len(X)/(sampling_rate)*Th):int(len(X) - len(X)/(sampling_rate)*Th)] = 0.001 + 0.001j
```

- `copy.deepcopy(X)` is required because plain assignment would make `Filtered_X` and `X` share the same array — modifying one would change the other.
- The slice from `Fs·smp / Fs · Th` (= `Th · N / Fs` samples) onward is replaced with a tiny complex number. This zeroes out every frequency bin above 1 kHz and below its mirror `Fs − 1 kHz`.
- Why `0.001 + 0.001j` and not `0`? Replacing with `0` would sharply truncate the spectrum and create ringing artefacts (Gibbs phenomenon). A tiny residual value avoids that.

```python
Filtered_amplitude = 20*np.log10(np.abs(Filtered_X) + 1e-10)
Filtered_phase = [np.arctan2(c.imag, c.real) for c in Filtered_X[0:int(len(X)/2)]]
```

Recompute amplitude and phase spectra for plotting.

```python
Filtered_x = np.fft.ifft(Filtered_X)
Filtered_x = Filtered_x.real
```

- `np.fft.ifft(...)` is the **inverse FFT** — converts the modified frequency-domain data back to a time-domain signal.
- The output is still complex (numerical noise leaves a tiny imaginary part), so `.real` discards it.

```python
Filtered_x = np.int16(0.9 * Filtered_x / max(Filtered_x) * 2**15)
```

- Normalises to the new max amplitude, scales to 0.9 of the available range to avoid clipping on playback, then converts to `int16` — the format the WAV writer expects.

```python
Filtered_wavfile = wave.open('sound/Filtered.wav', "w")
Filtered_wavfile.setnchannels(1)
Filtered_wavfile.setsampwidth(2)
Filtered_wavfile.setframerate(sampling_rate)
Filtered_wavfile.writeframes(Filtered_x.tobytes())
Filtered_wavfile.close()
```

- Opens a new WAV file in write mode.
- Mono, 16-bit, same sampling rate as the original.
- `.tobytes()` flattens the NumPy array to raw bytes that the WAV format can write.

The last plot overlays the original and filtered waveforms in the time domain: they look almost identical even though the audio sounds muffled, because the time-domain plot hides the loss of high frequencies. This is the key insight: features invisible in the time domain become obvious in the frequency domain.

## Key takeaways

- The FFT converts a time-domain signal into a list of (amplitude, phase) pairs — one per frequency bin.
- Amplitude on a dB scale matches human loudness perception.
- The phase component looks random but is necessary for an exact inverse transform.
- Filtering in the frequency domain (multiply, threshold, etc.) and then running `np.fft.ifft` is a powerful way to alter audio.
- Human vowels show harmonic structure: peaks at integer multiples of a fundamental frequency F0.