# SoundProcessing5.ipynb — Speech Recognition with OpenAI Whisper

The final notebook of the series steps away from classical signal processing and uses a deep-learning-based speech recognition system: **Whisper**, released by OpenAI.

The audio sample is `speech.wav` — a woman reading the opening sentence of the Japanese Wikipedia article on speech recognition: "音声認識は声がもつ情報をコンピュータに認識させるタスクの総称である。ヒトの音声認識と対比して自動音声認識とも呼ばれる。"

## Setup

```python
urllib.request.urlretrieve('...speech.wav...', './sound/speech.wav')
```

Downloads the speech sample.

```python
import IPython
IPython.display.Audio('sound/speech.wav')
```

Embeds an HTML5 audio player so you can listen to the file inside the notebook.

## Library installation

```python
#!pip install -q openai-whisper
```

- The leading `#` comments the line out because Whisper is already installed in the notebook's environment. Uncomment it on a fresh machine.
- `-q` (quiet) suppresses pip's chatty output, leaving only errors.
- `openai-whisper` is the official Whisper Python package.

## Transcription with the `base` model

```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("sound/speech.wav")
print(result["text"])
```

- `whisper.load_model("base")` downloads and caches the `base` multilingual Whisper model (~74 M parameters) on first use, then loads it into memory.
- `model.transcribe(path)` runs the full recognition pipeline: audio loading → resampling to 16 kHz → log-Mel spectrogram → encoder → decoder with cross-attention → token sampling. It returns a dictionary.
- `result["text"]` is the recognised transcript as a single string.

The notebook's actual run produced an error (`ModuleNotFoundError: No module named 'whisper'`) in this environment because Whisper isn't installed here, but the code is correct — it would work after running the commented-out `pip install` cell.

## Trying a larger model (`medium`)

```python
model = whisper.load_model("medium")
result = model.transcribe("sound/speech.wav")
print(result["text"])
```

- `medium` is roughly 10× larger (769 M parameters) and generally produces more accurate Japanese transcripts than `base`.
- Whisper's published model sizes are: `tiny`, `base`, `small`, `medium`, `large` (and the newer `large-v2`, `large-v3`). Each step is roughly an order of magnitude more compute.

## How Whisper works

The markdown cell summarises the architecture:

1. The raw waveform is converted to a **log-Mel spectrogram** — like the STFT in notebook 4, but with a Mel filterbank applied. The Mel scale mimics human hearing: fine frequency resolution at low frequencies, coarse at high frequencies.
2. The spectrogram is passed through a stack of **Transformer encoder** blocks to produce a sequence of audio features.
3. In parallel, the decoder generates the output text token-by-token. At each step it attends to both the previously generated tokens and (via **cross-attention**) the encoded audio features.
4. The output tokens are decoded back into the transcript string.

The same architecture is used for many other languages — the only thing that changes is the training data and the vocabulary.

## Key takeaways

- Whisper is an end-to-end model: feed it audio, get text back.
- The `base` and `medium` size variants give different speed/accuracy trade-offs.
- The internal representation is a **log-Mel spectrogram** — the same family of features introduced in notebooks 3 and 4.
- The decoder uses cross-attention to align audio features with text tokens, conceptually similar to neural machine translation.