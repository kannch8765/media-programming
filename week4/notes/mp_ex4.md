# mp_ex4.ipynb — Exercise: Find the Most Positive Sentence in *Gauche the Cellist*

This is the **assignment** notebook for week 4. The task is to take the first 10 sentences of Miyazawa Kenji's *Gauche the Cellist* (「セロ弾きのゴーシュ」), translate them into English with a pre-trained model, find the English sentence most strongly classified as `POSITIVE`, and print the corresponding original Japanese sentence.

## Setup cell — download the data and install libraries

```python
import urllib.request
import os
import sys

os.makedirs('text', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/serohikino_goshu_org.txt',
    './text/serohikino_goshu_org.txt'
)

!pip install transformers
!pip install fugashi
!pip install unidic-lite
!pip install sentencepiece
!pip install sacremoses
```

- Downloads the raw text of 「セロ弾きのゴーシュ」.
- Installs Hugging Face `transformers` and its Japanese-tokenisation dependencies.

## Loading and splitting the text into sentences

```python
infile = 'text/serohikino_goshu_org.txt'
texts = []
with open(infile, 'r', encoding='utf-8') as fi:
    for line in fi:
        texts.extend(line.rstrip('\n').replace('。', '。<EOS>').rstrip('<EOS>').split('<EOS>'))

texts[:10]
```

- The file is one long paragraph; the code splits it at every Japanese period `。`.
- `replace('。', '。<EOS>')` inserts a marker after each period, then `split('<EOS>')` produces a list of sentences.
- `rstrip('<EOS>')` removes a trailing marker so the last split doesn't create an empty string.
- `texts[:10]` is the list of the first 10 sentences you will work with.

## Task 1 — Translate the first 10 sentences to English

> Use the Japanese-to-English translation model `Helsinki-NLP/opus-mt-ja-en` from `TopicAnalysis4.ipynb`. Store the 10 translations in a list called `translated`, preserving order.

```python
from transformers import pipeline

translator = pipeline("translation", model='Helsinki-NLP/opus-mt-ja-en')

translated = []
for sentence in texts[:10]:
    result = translator(sentence)
    translated.append(result[0]['translation_text'])
```

- `pipeline("translation", model=...)` loads the seq2seq translation model and tokenizer together.
- `translator(sentence)` returns a list with one dict: `{'translation_text': ...}`.
- Keep the translations in the same order as the original sentences so Task 3 can map back.

Expected first 10 translations look like:

```python
["Gorsh was the guy who played celery in the city's active photo theater.",
 'But I had a reputation for not being very good at it.',
 "I wasn't very good at it, but in fact, I was the worst of all the other players, so I was always treated easily.",
 "Everyone who was too upset was circled into the dressing room and was practicing the first symphony for this town's concert.",
 'The trumpet is singing for life.',
 'Violet is also singing like a double wind.',
 'The clarinet also helps with the bowl.',
 'Gorsh, too, ties his mouth tightly together and looks like a plate in his eyes, and plays the music one more time as he looks at it.',
 'Suddenly, Kenji rang his hands.',
 'Everyone dropped the music quickly and tried to stop it.']
```

## Task 2 — Find the index of the most positive sentence

> Use the sentiment-analysis model `distilbert-base-uncased-finetuned-sst-2-english` from `TopicAnalysis4.ipynb`. Find which translated sentence is predicted `POSITIVE` with the highest score, and store its index in `maxid`.

```python
classifier = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')

maxid = 0
max_score = 0.0
for i, sentence in enumerate(translated):
    result = classifier(sentence)[0]
    if result['label'] == 'POSITIVE' and result['score'] > max_score:
        max_score = result['score']
        maxid = i
```

- `classifier(sentence)[0]` returns `{'label': 'POSITIVE'|'NEGATIVE', 'score': float}`.
- Track the highest positive score and its index.
- `maxid` is the variable the grader checks.

## Task 3 — Print the corresponding Japanese sentence

> Store the Japanese sentence at index `maxid` in `rslt` and print it.

```python
rslt = texts[maxid]
print(rslt)
```

- Because `translated[i]` is the translation of `texts[i]`, mapping back is a simple index lookup.
- `rslt` is the variable the grader checks.

## Key takeaways

- This assignment chains two pre-trained Hugging Face pipelines: **translation** (Japanese → English) and **sentiment analysis** (English).
- The hard part is data plumbing, not model training: split the Japanese text, keep indices aligned, translate, classify, and map back.
- `pipeline` hides tokenisation, model forward passes, and decoding — you only need to move text in and out.
- Always verify that intermediate outputs (`translated`, classification results) make sense before trusting the final `maxid`/`rslt`.
