# TopicAnalysis1.ipynb — Bag-of-Words Document Vectors

This notebook is the first in the "Topic Analysis" series. It introduces **Bag-of-Words (BoW)** — representing a document as a vector that counts how many times each vocabulary word appears. Word order is ignored; only the multiset of words matters.

The notebook walks through:

1. Building BoW vectors for 8 tiny Japanese sentences.
2. Computing **cosine similarity** between document vectors.
3. Extending the same idea to real novels: Miyazawa Kenji's *Night on the Galactic Railroad* (「銀河鉄道の夜」) and *Matasaburo of the Wind* (「風の又三郎」), and Dazai Osamu's *No Longer Human* (「人間失格」).

## Setup cell — download the data

```python
import urllib.request
import os
import zipfile

os.makedirs('text', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc4/mediaproc4-1.zip',
    './text/mediaproc4-1.zip'
)
with zipfile.ZipFile('text/mediaproc4-1.zip') as zip:
    zip.extractall('text/')
```

- Downloads a zip of pre-tokenised (wakati) novel files into `text/`.
- `os.makedirs('text', exist_ok=True)` creates the directory only if it doesn't already exist.
- `zipfile.ZipFile(path).extractall(target)` unpacks the archive.

## 1. Document vectors from Bag-of-Words

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

text = ['私 は 本 を 読む 。',
        '今日 は 晴天 だ 。',
        '私 は 私 だ 。',
        '本 は 本 で 本 だ 。',
        '私 は 本 を 読む 。',
        '私 本 を 読む 。',
        '私 は 本 を 読む 。 本 を 私 は 読む 。',
        '明日 雨 が 降る ！']

count = CountVectorizer(token_pattern=r'[^\s]+')
bow = count.fit_transform(text)

print('語彙サイズ:', len(count.get_feature_names_out()))
print(sorted(count.vocabulary_.items(), key=lambda x:x[1]))

vec = bow.toarray()
for i in range(len(text)):
    print(text[i], ':\t', vec[i])
```

- `CountVectorizer` turns a list of strings into a document-term count matrix.
- `token_pattern=r'[^\s]+'` tells scikit-learn to treat any non-space sequence as one token. This is essential because the input is already 分かち書き (space-separated).
- `fit_transform(text)` learns the vocabulary from the corpus and returns the BoW matrix in one step.
- `count.vocabulary_` maps each word to its column index; `get_feature_names_out()` returns the words in column order.
- `bow.toarray()` converts the sparse matrix to a dense NumPy array so we can inspect it.

The vocabulary has 15 unique words, so every document becomes a 15-dimensional vector. The value at each dimension is the raw count of that word in the document.

## 2. Cosine similarity between vectors

```python
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

for i in range(1, len(text)):
    sim = cos_sim(vec[0], vec[i])
    print('「', text[0], '」と「', text[i], '」の類似度:\t', round(sim, 3))
```

- Cosine similarity measures the cosine of the angle between two vectors:

  $$\cos\text{_sim}(V_1, V_2) = \frac{V_1 \cdot V_2}{|V_1| |V_2|}$$

- Because all counts are non-negative, document vectors live in the positive orthant and cosine similarity ranges from **0 to 1** (not -1 to 1).
- Identical documents score 1.0, even if one is exactly twice as long: doubling every count scales the vector but does not change its angle.
- Sentences that share many words score high; sentences with only function words in common score low.

## 3. Similarity between novels

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

titles = ['銀河鉄道の夜', '風の又三郎', '人間失格']
files = ['text/gingatetsudonoyoru_wakati.txt',
         'text/kazenomatasaburo_wakati.txt',
         'text/ningenshikkaku_wakati.txt']

wakati_all = []
for i in range(len(files)):
    with open(files[i], 'r', encoding='utf-8') as f:
        wakati_all.append(f.read().replace('\n', ' '))

novel_count = CountVectorizer(token_pattern=r'[^\s]+')
novel_bow = novel_count.fit_transform(wakati_all)
novel_vec = novel_bow.toarray()
```

- Each novel is read as one long space-separated string (newlines replaced by spaces).
- `CountVectorizer` builds a single vocabulary across all three novels and returns one vector per novel.

```python
print('語彙サイズ:', len(novel_count.get_feature_names_out()))

novel_sim = []
for i in range(len(files)):
    novel_sim.append([])
    for j in range(len(files)):
        novel_sim[i].append(cos_sim(novel_vec[i], novel_vec[j]))
print('「銀河鉄道の夜」,「風の又三郎」,「人間失格」')
print(np.array(novel_sim))
```

- The pairwise cosine matrix shows which novels are "close" in vocabulary space.
- The two Miyazawa novels are typically very similar to each other (same author, same style/fantasy register), while Dazai's novel is somewhat farther away.

## Key takeaways

- **Bag-of-Words** ignores grammar and word order; a document is just a count vector over a fixed vocabulary.
- `CountVectorizer(token_pattern=r'[^\s]+')` is the right choice when the input is already 分かち書き.
- **Cosine similarity** compares the *direction* of two vectors, so it is not fooled by document length.
- Identical documents score 1.0, and a document scores 1.0 with any positive scalar multiple of itself.
- Real documents (novels) cluster by author/style in BoW space, making this a surprisingly useful baseline for text similarity.
