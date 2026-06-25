# TextProcessing3.ipynb — Statistical n-gram Language Models

This notebook builds a **statistical language model** over the 147 works of Miyazawa Kenji, using Janome pre-tokenisation plus NLTK's n-gram infrastructure. The model is then used to:

1. Look up the most likely next word given a 1- or 2-word context.
2. Generate random sentences by sampling from the n-gram distribution.
3. Score the "Miyazawa-likeness" of any input sentence by computing its **sentence probability** `P(W)`.

The same example sentence / word appears throughout: 「ジヨバンニ」 (note: pre-war katakana for "ジョバンニ").

## Setup

```python
import urllib.request
import os
import sys

os.makedirs('text', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/miyazawa_wakati.txt',
    './text/miyazawa_wakati.txt'
)

# Colab only:
if 'google.colab' in sys.modules:
    !pip install --upgrade nltk
```

- Downloads `miyazawa_wakati.txt`, the pre-tokenised concatenation of all 147 Aozora Bunko texts by Miyazawa Kenji (one sentence per line, Janome-separated by spaces).
- Colab-only NLTK upgrade, then **restart runtime** so the new version is actually used.

## 1. Preparation — split into words with sentence markers

```python
file = 'text/miyazawa_wakati.txt'

words = [('<BOP> ' + l + ' <EOP>').split() for l in open(file, 'r', encoding='utf-8').readlines()]

print(words[:2])
```

- Each line of the source file becomes one element of `words` (a list of lists).
- `<BOP>` (Beginning Of Paragraph) and `<EOP>` (End Of Paragraph) are inserted at the start and end of each sentence. These pseudo-tokens:
  - give the model a known anchor for sentence boundaries,
  - let it learn what words tend to start or end sentences,
  - make per-line n-grams work without special-casing.
- The construction `('<BOP> ' + l + ' <EOP>').split()` builds the augmented string and splits on whitespace in one go.

## 2. n-gram generation

```python
from nltk.util import ngrams

text_unigrams = [ngrams(word, 1) for word in words]   # uni-gram
text_bigrams  = [ngrams(word, 2) for word in words]   # bi-gram
text_trigrams = [ngrams(word, 3) for word in words]   # tri-gram

print('unigram', [x for x in text_unigrams[0]])
print('bigram',  [x for x in text_bigrams[0]])
print('trigram', [x for x in text_trigrams[0]])
```

- `ngrams(sequence, n)` is NLTK's generator that yields overlapping length-`n` tuples from a sequence.
  - `n=1` → **uni-gram**: each word alone
  - `n=2` → **bi-gram**: every pair of consecutive words
  - `n=3` → **tri-gram**: every triple of consecutive words
- The output of `ngrams` is an **iterator**, not a list — exhaust it once and it's empty. If you need to use the same n-grams twice (next cell does!), recompute them.

### 2.1 Counting n-gram occurrences

```python
from nltk.util import ngrams
from nltk.lm import NgramCounter

text_unigrams = [ngrams(word, 1) for word in words]
text_bigrams  = [ngrams(word, 2) for word in words]
text_trigrams = [ngrams(word, 3) for word in words]

ngram_counts = NgramCounter(text_unigrams + text_bigrams + text_trigrams)
```

- `NgramCounter` is NLTK's pre-built n-gram frequency counter. Pass it the (re-generated) iterators.
- It indexes every n-gram under a unified key, so you can query any n:

  | Query | Returns |
  | --- | --- |
  | `ngram_counts['ジヨバンニ']` | how often 「ジヨバンニ」 appears (uni-gram count) |
  | `ngram_counts[['ジヨバンニ']]` | dict of `next_word → count` for bi-grams starting with 「ジヨバンニ」 |
  | `ngram_counts[['ジヨバンニ','は']]` | dict of `next_word → count` for tri-grams starting with 「ジヨバンニ は」 |

### 2.2 Inspecting the counts

```python
print('ジヨバンニ: ' + str(ngram_counts['ジヨバンニ']))

print('ジヨバンニ')
for word, count in sorted(ngram_counts[['ジヨバンニ']].items(), key=lambda x:x[1], reverse=True):
    print('->\t{:s}: {:d}'.format(word, count))

print('ジヨバンニ は')
for word, count in sorted(ngram_counts[['ジヨバンニ','は']].items(), key=lambda x:x[1], reverse=True):
    print('->\t{:s}: {:d}'.format(word, count))
```

- The three blocks print:
  1. The **uni-gram count** of 「ジヨバンニ」 in the whole corpus.
  2. The **bi-gram successors** of 「ジヨバンニ」, sorted by count.
  3. The **tri-gram successors** of 「ジヨバンニ は」, sorted by count.
- For example, in the Miyazawa corpus, 「ジヨバンニ は → 、」 occurs 33 times. Combined with the fact that 「ジヨバンニ は」 itself occurs 123 times, that gives a **tri-gram probability** of 33/123 ≈ 0.268.

## 3. Training the language model

```python
from nltk.lm import Vocabulary
from nltk.lm.models import MLE
from nltk.util import ngrams

vocab = Vocabulary([item for sublist in words for item in sublist])
print('Vocabulary size: ' + str(len(vocab)))

text_trigrams = [ngrams(word, 3) for word in words]

n = 3
lm = MLE(order=n, vocabulary=vocab)
lm.fit(text_trigrams)
```

- **Vocabulary** flattens the 2-D `words` into a 1-D stream (list comprehension) and counts every distinct token. The size is the number of unique word types in the corpus.
- **MLE** (Maximum Likelihood Estimation) is the simplest n-gram language model — the probability of an n-gram is just its relative frequency:

  $$p(w_i \mid w_{i-n+1},\dots,w_{i-1}) = \frac{\mathrm{count}(w_{i-n+1},\dots,w_i)}{\mathrm{count}(w_{i-n+1},\dots,w_{i-1})}$$

- `lm.fit(corpus_iterators)` populates the model's count tables from the (re-generated) trigram iterators.
- Training is computationally modest for tri-grams over ~150 short novels, but the notebook warns that it "takes some time" — be patient on first run.

## 4.1 Querying the trained model

```python
context = ['ジヨバンニ', 'は']
print(context)

prob_list = [(word, lm.score(word, context)) for word
             in lm.context_counts(lm.vocab.lookup(context))]
prob_list.sort(key=lambda x: x[1], reverse=True)

sum_prob = 0.0
for word, prob in prob_list:
    print('\t{:s}: {:f}'.format(word, prob))
    sum_prob += prob

print('ある単語列に続くtri-gram確率をすべて足したら', sum_prob)
```

- `lm.vocab.lookup(context)` maps any out-of-vocabulary word to the `<UNK>` pseudo-token — essential because NLTK's MLE model will only score words it has seen in training.
- `lm.context_counts(context)` returns the set of words observed to follow that context in training, with their raw counts.
- `lm.score(word, context)` returns the conditional probability `P(word | context)` (the tri-gram probability).
- Sum of all successor probabilities over a fixed context is **1.0** (up to floating-point error) — confirming that this is a properly normalised distribution. That is what makes it a **conditional probability distribution** rather than just a frequency.

## 4.2 Random sentence generation

```python
context = ['ジヨバンニ', 'は']

for i in range(0, 100):
    w = lm.generate(text_seed=context)
    context.append(w)
    print(context)

    if '。' == w or '<EOP>' == w:
        break
```

- `lm.generate(text_seed=context)` samples one word from the tri-gram distribution conditioned on the **last 2 words** of `text_seed` (because the model is tri-gram).
- Each iteration:
  1. The newly sampled word is appended to `context`.
  2. The whole prefix is printed so you can watch the sentence grow.
  3. The loop stops when 「。」 or `<EOP>` is drawn.
- The 100-iteration cap is a safety net — in practice, the loop terminates within ~30 words for this corpus.
- The notebook calls this output **人工無能** ("artificial incompetence") — a tongue-in-cheek reference to the classic ELIZA-style chatbots that operate by stringing probable words together without any real understanding.

## 5. Sentence probability P(W)

The notebook's mathematical core is computing how likely a sentence is under the trained model:

$$P(W) = \prod_{i=0}^{n+1} p(w_i \mid w_{i-n+1}, \dots, w_{i-1})$$

where `w_0` is `<BOP>` (sentence start) and `w_{n+1}` is `<EOP>` (sentence end).

```python
line = 'ジヨバンニ は 何 げ なく 答え まし た 。'
# line = '何 げ なく ジヨバンニ は 答え まし た 。'

words2 = line.split()

n = 3
probability = 1.0
for ngram in ngrams(words2, n):
    prob = lm.score(lm.vocab.lookup(ngram[-1]),
                    lm.vocab.lookup(ngram[:-1]))
    print(ngram[:-1], ngram[-1], ':\t', prob)

    prob = max(prob, 1e-8)  # avoid zero collapse

    probability *= prob

print(probability)
```

- The loop multiplies the tri-gram probabilities along the sentence. Note that the multiplication makes the final `probability` extraordinarily small — that's expected, because it's a product of dozens of fractions each < 1.
- The `max(prob, 1e-8)` step is **smoothing against zero**: if any tri-gram was never seen in training, its raw probability is 0, and multiplying by 0 annihilates the whole product. Replacing zero with `1e-8` keeps the score from collapsing. This is a hack — proper smoothing techniques (Laplace, Kneser-Ney) are the textbook fix — but for a relative comparison between two sentences, the hack works.

### The notebook's punchline

Two sentences with the same words but different orders are scored:

- 「何 げ なく ジヨバンニ は 答え まし た 。」 → `P = 6.659 × 10⁻²⁵`
- 「ジヨバンニ は 何 げ なく 答え まし た 。」 → `P = 3.125 × 10⁻⁵`

The second is **20 orders of magnitude more probable** — because Miyazawa's writing style routinely places 「ジヨバンニ は」 at the start of sentences. The model has learned this from the corpus, and the probability product reflects it.

## Key takeaways

- An **n-gram language model** assigns probabilities to word sequences by counting how often each n-gram appeared in training text.
- The simplest estimator is **MLE** (relative frequency). Real systems use **smoothing** to handle never-seen n-grams.
- NLTK provides all the pieces: `ngrams` to enumerate, `NgramCounter` to count, `Vocabulary` for type-tracking, `MLE` for the model.
- Once trained, the model can:
  1. **Predict** the next word given a context (`lm.score`).
  2. **Generate** text by sampling sequentially (`lm.generate`).
  3. **Score** sentences for likelihood (`P(W) = product of n-gram probabilities`).
- One unobserved n-gram can zero out the whole sentence probability — always floor low-probability events, or use a proper smoothing method.
- This is the conceptual ancestor of every modern LLM: today's transformers are essentially very-deep, very-wide n-gram models trained on much larger corpora.
