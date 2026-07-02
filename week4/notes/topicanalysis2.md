# TopicAnalysis2.ipynb — Important-Word Extraction with tf-idf

This notebook introduces **tf-idf** (term frequency–inverse document frequency), a way to weight words so that the most characteristic words of a document stand out. It then applies tf-idf to six categories of Wikipedia articles and uses the resulting vectors for category-to-category similarity and simple document classification.

## Setup cell — download the data and install the font

```python
import urllib.request
import os
import zipfile
import sys

os.makedirs('fig', exist_ok=True)
os.makedirs('text', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc4/mediaproc4-2.zip',
    './text/mediaproc4-2.zip'
)
with zipfile.ZipFile('text/mediaproc4-2.zip') as zipf:
    zipf.extractall('text/')

if 'google.colab' in sys.modules:
    !apt-get -y install fonts-ipafont-gothic
    !rm /root/.cache/matplotlib/*.json
```

- Downloads `mediaproc4-2.zip`, which contains pre-tokenised Wikipedia articles grouped into six categories.
- Installs IPAfont Gothic on Colab so Japanese text renders in plots. A runtime restart is required after font installation.

## 1. Loading the Wikipedia articles

```python
import json

Categories = ['animal', 'art', 'economy', 'law', 'plant', 'politics']

with open('text/wikipedia_wakati.json', 'r', encoding='utf-8') as fi:
    wiki = json.load(fi)

print([len(wiki[cate]) for cate in Categories])

for doc in wiki['animal']:
    print('docID:', doc)
    print('title:', wiki['animal'][doc]['title'])
    print('url:', wiki['animal'][doc]['url'])
    print('wakati:', wiki['animal'][doc]['wakati'])
    break
```

- The JSON is a nested dictionary: `wiki[category][doc_id]` contains `url`, `title`, and `wakati` (space-separated tokens).
- Six categories are analysed: animal, art, economy, law, plant, politics.
- Each category holds roughly 1 MB of tokenised article text.

## 2. What tf-idf measures

Tf-idf combines two ideas:

- **tf (term frequency)**: a word that appears many times in a document is probably important *to that document*.
- **idf (inverse document frequency)**: a word that appears in almost every document is not a useful discriminator.

$$\text{tf-idf}(w, d) = \text{tf}(w, d) \times \text{idf}(w)$$

$$\text{tf}(w, d) = \frac{\text{count of } w \text{ in } d}{\text{total words in } d}$$

$$\text{idf}(w) = \log \frac{\text{total documents}}{\text{documents containing } w}$$

The notebook uses scikit-learn's `TfidfVectorizer` rather than hand-rolling the formula.

## 3. Building the corpus and computing tf-idf

```python
corpus = [
    ''.join(
        line.rstrip('\n')
        for item in wiki[cate].values()
        for line in item['wakati']
    )
    for cate in Categories
]

for c in corpus:
    print(c[0:50])
print([len(v) for v in corpus])
```

- `corpus` has one very long string per category, formed by concatenating every article's wakati text.
- `''.join(...)` is used because the wakati text already contains spaces; no extra delimiter is needed.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000, max_df=5, min_df=3)
X = vectorizer.fit_transform(corpus)

print(X.shape)
```

- `max_features=10000` keeps only the 10,000 most frequent terms, limiting memory and vector dimension.
- `max_df=5` drops terms that appear in more than 5 of the 6 categories (too common to be discriminative).
- `min_df=3` drops terms that appear in fewer than 3 categories (too rare to be compared across documents).
- The result `X` is a 6 × up-to-10,000 sparse tf-idf matrix.

## 4. Inspecting tf-idf scores

### 4.1 Vocabulary and per-word scores

```python
feature_names = vectorizer.get_feature_names_out()
for i in range(1000, 1010):
    print(i, ':\t', feature_names[i])
```

- `feature_names` gives the word assigned to each column of `X`.

```python
import numpy as np

for w in ['裁判官', '画家', '生育', '細胞', '融資', '衆議院']:
    ID = np.where(feature_names == w)[0][0]
    print('ID:', ID, '単語：', feature_names[ID])
    for cate_n in range(len(Categories)):
        print('{0:>10}: {1:.4f}'.format(Categories[cate_n], float(X[cate_n, ID])))
    print()
```

- For each query word, this prints its tf-idf value in each category.
- Characteristic words (e.g. 裁判官 in law, 画家 in art) have high values in their own category and low values elsewhere.

### 4.2 Top words per category

```python
dic = {}
for cate in Categories:
    pair = dict(zip(feature_names, X[Categories.index(cate), :].toarray()[0]))
    dic[cate] = [(x, pair[x]) for x in sorted(pair, key=lambda x: -pair[x])]

for cate in Categories:
    print(cate, dic[cate][:20])
    print()
```

- Builds a dictionary `dic[category] = [(word, tf-idf), ...]` sorted from highest to lowest tf-idf.
- Printing the top 20 words quickly reveals the vocabulary fingerprint of each category.

### 4.3 Word cloud from tf-idf weights

```python
%matplotlib inline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

cate = 'art'
wordfreq = {w[0]: w[1] for w in dic[cate]}

wc = WordCloud(
    background_color="white",
    font_path='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
    width=640, height=480
).generate_from_frequencies(wordfreq)

plt.figure(figsize=(12, 9))
plt.imshow(wc)
plt.axis("off")
plt.savefig('fig/TopicAnalysis1-1.png')
```

- Unlike the week 3 word cloud, which counted raw frequencies, this one passes tf-idf weights.
- `generate_from_frequencies` takes any `word → weight` dictionary; larger weights are drawn bigger.
- Change `cate` to explore other categories.

## 5. Similarity and classification with tf-idf vectors

### 5.1 Category-to-category cosine similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(X)
print(Categories)
print(similarity_matrix)
```

- `cosine_similarity(X)` computes the pairwise similarity of the six tf-idf document vectors.
- Categories with overlapping vocabulary (e.g. law and politics, animal and plant) tend to score higher.

### 5.2 Classifying an unseen article

```python
with open('text/wikipedia_sample.json', 'r', encoding='utf-8') as fi:
    wiki_sample = json.load(fi)
```

```python
sample = wiki_sample['art']['3912545']['wakati'].replace('\n', '')
sample_tfidf = vectorizer.transform([sample])
print(sample_tfidf.shape)
```

- `vectorizer.transform([sample])` converts a new document into the *same* 10,000-dimensional tf-idf space. Do **not** call `fit_transform` again — that would rebuild the vocabulary.

```python
for x in sample_tfidf[0].nonzero()[1]:
    print(feature_names[x], sample_tfidf[0, x])
```

- Lists only the vocabulary words that the new article shares with the training corpus and their tf-idf scores.

```python
sample_sim_matrix = cosine_similarity(sample_tfidf, X)[0]

plt.bar(range(len(sample_sim_matrix)), sample_sim_matrix.flatten(), tick_label=Categories)
print(Categories)
print(sample_sim_matrix)
plt.savefig('fig/TopicAnalysis2-2.png')
```

- The unseen article is classified as the category whose tf-idf vector is most cosine-similar.
- Because law and politics are already close in the training corpus, their bars can be close, making discrimination harder.

## Key takeaways

- **tf-idf** weights words by how often they appear in a document *and* how rarely they appear across documents.
- `TfidfVectorizer` handles tokenisation, counting, weighting, and optional frequency pruning in one object.
- `max_features`, `max_df`, and `min_df` are the main levers for controlling speed, memory, and discriminative power.
- Once a `TfidfVectorizer` is fitted, `transform` maps new documents into the same vector space for comparison or classification.
- tf-idf vectors plus cosine similarity give a simple but effective baseline for topic similarity and nearest-category classification.
