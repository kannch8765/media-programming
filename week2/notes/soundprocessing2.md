# SoundProcessing2.ipynb — Sampling Rate and Bit Depth

This notebook explores the two main parameters that determine how faithfully analogue sound is captured into digital form: the **sampling rate** (how many times per second we measure the waveform) and the **bit depth** (how precisely we record each individual measurement).

The two piano WAV files used are:
- `2-Piano-A4-16bit-44100Hz.wav` — 16-bit, 44.1 kHz
- `2-Piano-A4-8bit-44100Hz.wav` — 8-bit, 44.1 kHz

## Setup

```python
urllib.request.urlretrieve(...16bit... , './sound/2-Piano-A4-16bit-44100Hz.wav')
urllib.request.urlretrieve(...8bit...  , './sound/2-Piano-A4-8bit-44100Hz.wav')
```

Downloads the two WAV files into `sound/`. The earlier notebooks have already created the `fig` and `sound` folders, so the `mkdir` warning is harmless.

## 1.1 Sampling rate vs. sound frequency

```python
fig = plt.figure(figsize=(7, 10))
WaveFreq = 5

for sampling_rate in [100, 90, 80, 70, ...]:
    ax = plt.subplot(7, 3, num)
    rad = [i*WaveFreq*(2*math.pi)/sampling_rate for i in range(sampling_rate)]
    data = [math.sin(i) for i in rad]
    x = np.arange(0, len(data)/sampling_rate, 1.0/sampling_rate)
    ax.plot(x, data)
    ax.set_xlim(0, 1.0)
    ax.grid()
    plt.title('sampling rate: ' + str(sampling_rate))
```

- The loop produces a 5 Hz sine wave sampled at progressively lower rates, from 100 Hz down to 2 Hz.
- `rad[i] = 2π · WaveFreq · i / sampling_rate` — this is the phase at sample index `i`. Dividing by `sampling_rate` is what makes it the discrete phase.
- `data[i] = sin(rad[i])` — converts each phase into an amplitude value.
- `np.arange(0, duration, 1/sampling_rate)` builds the time axis: start at 0 s, end at 1 s, stepping by exactly one sample.
- `plt.subplot(7, 3, num)` places each plot in a 7×3 grid of subplots. `num` increments each loop iteration.

The intended observation: once the sampling rate drops below `2 × WaveFreq` (i.e. below 10 Hz for a 5 Hz signal), the resulting plot no longer shows 5 peaks — the original waveform has been lost.

## 1.2 Aliasing

```python
sampling_rate = 50
WaveFreq1 = 5
WaveFreq2 = sampling_rate - WaveFreq1  # = 45

rad1 = [i*WaveFreq1*(2*math.pi)/sampling_rate for i in range(sampling_rate)]
y1 = [math.sin(i) for i in rad1]
rad2 = [i*WaveFreq2*(2*math.pi)/sampling_rate for i in range(sampling_rate)]
y2 = [math.sin(i) for i in rad2]

ax0.plot(y1, marker="o")
ax1.plot(y2, marker="o")
```

- Two sine waves are generated: 5 Hz and 45 Hz.
- Both are sampled at 50 Hz.
- Even though the underlying frequencies are different, the sampled points fall in exactly the same places, so the two plots look identical.

This is **aliasing** — when the sampling rate is too low, a high-frequency signal is mistaken for a low-frequency one. The general rule: at sampling rate `F`, a frequency `f` looks identical to `F − f`.

## 1.3 Sampling theorem

The text introduces the **Nyquist frequency** (`F/2`). To capture any frequency `f_max` without losing information (or creating aliasing artefacts), the sampling rate must satisfy `F > 2 · f_max`. Two practical rules:

1. Sample at more than `2 · f_max`.
2. Apply an analogue low-pass filter **before** sampling to remove anything above `f_max`.

## 2.1 Quantisation error (bit depth)

```python
wavfile_16bit = wave.open("sound/2-Piano-A4-16bit-44100Hz.wav", "rb")
wavfile_8bit  = wave.open("sound/2-Piano-A4-8bit-44100Hz.wav",  "rb")

x_16bit = wavfile_16bit.readframes(wavfile_16bit.getnframes())
x_16bit = [float(i) for i in np.frombuffer(x_16bit, dtype="int16")]
x_16bit = x_16bit / np.max(np.abs(x_16bit))
wavfile_16bit.close()

x_8bit = wavfile_8bit.readframes(wavfile_8bit.getnframes())
x_8bit = [float(i) - 128 for i in np.frombuffer(x_8bit, dtype="uint8")]
x_8bit = x_8bit / np.max(np.abs(x_8bit))
wavfile_8bit.close()
```

- 16-bit WAV stores signed samples in `int16`, range −32 768 … 32 767.
- 8-bit WAV stores unsigned samples in `uint8`, range 0 … 255 — but audio amplitudes are signed, so `-128` shifts the zero-line to the middle.
- Dividing by the maximum absolute amplitude normalises both signals to `[-1, 1]`.

The plots overlay the two waveforms:
- On the full-scale plot they look almost identical.
- Zoomed into a small range of samples, the 8-bit trace jumps in clearly visible discrete steps (256 levels), while the 16-bit trace is smooth (65 536 levels).

This discreteness is the **quantisation error** — the difference between the true analogue value and the nearest representable digital value. It appears as a hiss-like noise in the audio.

## 2.2 Dynamic range

The text walks through the formula for sound pressure level in decibels:

`L_p = 20 · log10(p / p_0)`

with `p_0 = 1` for relative measurements.

For each bit depth, the dynamic range is roughly:
- 8-bit: `20 · log10(256) + 1.76 ≈ 49.92 dB`
- 16-bit: `20 · log10(65536) + 1.76 ≈ 98.09 dB`
- 24-bit: `20 · log10(16777216) + 1.76 ≈ 146.25 dB`

(The +1.76 dB correction accounts for the fact that quantisation noise is a triangular wave, whose RMS differs from a sine wave by `√(3/2) ≈ 1.76 dB`.)

Human hearing covers about 120 dB, so 16-bit may still leak faint quantisation noise into silence, which is why 24-bit is preferred for high-quality recording.

## 2.3 Microphone volume

The final section explains practical advice: with a fixed bit depth, the microphone gain should be set so the loudest expected sound uses most — but not all — of the available levels. If the gain is too low, you waste resolution (effectively recording in fewer bits). If it is too high, you get **clipping** (`音割れ`): samples exceeding the max value are pinned to the max, distorting the waveform into a flat top.

## Key takeaways

- Sampling rate sets the highest frequency you can faithfully capture (Nyquist).
- Bit depth sets the dynamic range (the ratio between quietest and loudest representable sound).
- Insufficient sampling rate ⇒ aliasing; insufficient bit depth ⇒ quantisation noise.
- Both can be avoided by choosing parameters based on the signal you intend to record.