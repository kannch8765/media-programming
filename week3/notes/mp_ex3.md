# mp_ex3_new.ipynb — Exercise: Generate a Yosano Akiko-style sentence

This is the **assignment** notebook for week 3. The task is to build a tri-gram language model from Yosano Akiko's translation of the *Tale of Genji* (「源氏物語 桐壺」) and use it to generate a sentence starting with 「帝は」.

The original instructions are in Japanese; the notes below give a plain-English summary plus a walkthrough of the boilerplate code and what you need to write.

## Setup

```python
import urllib.request
import os
import sys

os.makedirs('text', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/gakumonno_susume_org.txt',
    './text/gakumonno_susume_org.txt'
)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/genjimonogatari_org.txt',
    './text/genjimonogatari_org.txt'
)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/ningen_shikkaku_org.txt',
    './text/ningen_shikkaku_org.txt'
)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/wagahaiwa_nekodearu_org.txt',
    './text/wagahaiwa_nekodearu_org.txt'
)

# Colab only:
if 'google.colab' in sys.modules:
    !pip install janome
```

- Downloads the raw text files for the assignment (「源氏物語 桐壺」) plus three "bonus" texts:
  - 「学問のすすめ」 (Fukuzawa Yukichi)
  - 「人間失格」 (Dazai Osamu)
  - 「吾輩は猫である」 (Natsume Soseki)
- Install `janome` on Colab; on local machines it should already be present from notebook 1.

## Task 1 — Produce a word-separated (分かち書き) version of Genji

> Open `text/genjimonogatari_org.txt`, run it through Janome, and write a half-space-separated, line-preserved version to `text/genjimonogatari_wakati.txt`.

The starter code already prepares the `Tokenizer` (note: `t` must be created only once):

```python
# -*- coding:utf-8 -*-
from janome.tokenizer import Tokenizer

t = Tokenizer()

infile  = './text/genjimonogatari_org.txt'
outfile = './text/genjimonogatari_wakati.txt'
```

What you need to add:

```python
with open(infile, 'r', encoding='utf-8') as fi, \
     open(outfile, 'w', encoding='utf-8') as fo:
    for line in fi:
        tokens = [token.surface for token in t.tokenize(line, wakati=True)]
        fo.write(' '.join(tokens) + '\n')
```

- The `wakati=True` flag (introduced in notebook 1) returns just the surface forms as a list, so the loop body stays short.
- `' '.join(tokens)` rebuilds the line with half-width spaces between tokens — the canonical 分かち書き form.
- Writing `'\n'` after each line **preserves the original line breaks**, as the instructions require.

### Sanity check

```python
with open(outfile, 'r', encoding='utf-8') as fi:
    print(fi.read(30))
```

Should print:

```
どの 天皇 様 の 御代 で あっ た か 、 女御 とか
```

If you see `□□□□` (tofu boxes), check that:
1. The file encoding is `utf-8` end-to-end (read and write).
2. `wakati=True` is being passed.
3. The original text file downloaded correctly.

## Task 2 — Train the tri-gram model and generate sentences

> Train a tri-gram language model on `text/genjimonogatari_wakati.txt`. Variable name must be `lm`. Each line's tokens should be wrapped with `<BOP>` and `<EOP>` before training.

What you need to add (mirrors the pipeline from notebook 3):

```python
from nltk.lm import Vocabulary
from nltk.lm.models import MLE
from nltk.util import ngrams

file = './text/genjimonogatari_wakati.txt'
words = [('<BOP> ' + l + ' <EOP>').split() for l in open(file, 'r', encoding='utf-8').readlines()]

vocab = Vocabulary([item for sublist in words for item in sublist])

text_trigrams = [ngrams(word, 3) for word in words]
lm = MLE(order=3, vocabulary=vocab)
lm.fit(text_trigrams)
```

Same as notebook 3:
- `'<BOP> ' + l + ' <EOP>'` wraps each line.
- `Vocabulary(...)` flattens and counts unique tokens.
- `MLE(order=3, ...)` builds the tri-gram model; `lm.fit(...)` populates its tables.

### Generating 10 sentences starting with 「帝は」

> The generated sentence must start with 「帝は」, end at one of 「。」, 「」」, or 「<EOS>」, and be in **normal Japanese text** (not a token list, not a 分かち書き string).

What you need to add:

```python
for _ in range(10):
    context = ['帝', 'は']
    sentence = context.copy()
    while True:
        w = lm.generate(text_seed=context)
        sentence.append(w)
        context = sentence[-2:]
        if w in ['。', '」', '<EOP>']:
            break
    print(''.join(sentence) + '\n')
```

- `lm.generate(text_seed=context)` samples the next word using only the **last 2 words** of the seed (because the model is tri-gram).
- After each new word, `context = sentence[-2:]` slides the window forward.
- `''.join(sentence)` reconstructs the natural Japanese form (no spaces). This is the required output format — the grader will compare it to Japanese text, not a token list.
- Stop conditions: any of 「。」, 「」」, or `<EOP>` ends the sentence. (Adding `<EOS>` lets you end without reaching a punctuation mark.)
- Loop 10 times to produce 10 sentences.

## A note on smoothing (optional improvement)

The base MLE model assigns probability **0** to any tri-gram it never saw — so the generated sentence can stall if the chosen context was rare. You have two options:

1. **Soft smoothing in the generator** (cheap): if `lm.generate` returns `<UNK>` or stalls, drop the oldest word from the seed and retry.
2. **Pre-fit smoothing**: pass `lr` (learning rate) and proper discount parameters when constructing `lm` — NLTK supports interpolated and modified Kneser-Ney models:

   ```python
   from nltk.lm.models import InterpolatedLanguageModel
   lm = InterpolatedLanguageModel(order=3, vocabulary=vocab)
   ```

The notebook doesn't require either — the basic MLE generator works fine on Genji because the corpus is large enough — but it's a natural extension if you have time.

## Bonus — try other corpora

The downloaded "bonus" texts can be substituted into the same pipeline to compare writing styles:

| File | Author |
| --- | --- |
| `gakumonno_susume_org.txt`   | 福沢諭吉 「学問のすすめ」 (Fukuzawa Yukichi) |
| `wagahaiwa_nekodearu_org.txt` | 夏目漱石 「吾輩は猫である」 (Natsume Soseki) |
| `ningen_shikkaku_org.txt`     | 太宰治 「人間失格」 (Dazai Osamu) |

For each:
1. Replace `genjimonogatari_org.txt` with the new file in Task 1 to produce a `_wakati.txt`.
2. Re-run Task 2's training code on the new `_wakati.txt`.
3. Compare the generated 「帝は」-starting sentence to Yosano Akiko's output — does the style leak through?

> **Important**: per the assignment instructions, delete the bonus-experiment cells before submitting the notebook. The grader only wants the Genji implementation.

## Key takeaways

- This assignment is a **pipeline**: text → 分かち書き → n-gram training → next-word sampling → sentence assembly. Each stage is a one-line addition to the boilerplate.
- The pipeline reuses every concept from notebooks 1 (Janome) and 3 (NLTK n-grams + MLE).
- The most common mistake is forgetting `<BOP>`/`<EOP>` wrapping — without them the model has no idea where sentences start and end.
- Another common mistake is returning `sentence` as a Python list — the grader expects a plain Japanese string (use `''.join`).
- Generating **natural-looking** Japanese with a tri-gram model is genuinely possible on a single-author corpus: the vocabulary is small and the syntax is highly patterned, so the probabilities concentrate on plausible continuations.
