# TopicAnalysis3.ipynb — Word2vec Word Embeddings

This notebook introduces **Word2vec**, a neural-network-based method for turning words into dense vectors. The key idea: words that appear in similar contexts end up with similar vectors, and the vectors support arithmetic like `king - man + woman ≈ queen`.

The notebook trains a Word2vec model on 2,101 tokenised Japanese Wikipedia articles (animal, plant, politics, economy, law, art) and then explores similar words, vector arithmetic, PCA visualisation, and hierarchical clustering.

## Setup cell — download the data and install libraries

```python
import urllib.request
import os
import sys

os.makedirs('text', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('fig', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc4/wiki_wakati.txt',
    './text/wiki_wakati.txt'
)

if 'google.colab' in sys.modules:
    !pip install japanize-matplotlib
    !pip install gensim
```

- Downloads `wiki_wakati.txt`, a single file of tokenised Wikipedia sentences.
- Installs `japanize-matplotlib` (Japanese font support) and `gensim` (Word2vec implementation) on Colab.

## 1. Training the Word2vec model

```python
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text/wiki_wakati.txt')

model = word2vec.Word2Vec(
    sentences,
    vector_size=200,   # embedding dimension
    min_count=20,      # ignore words seen fewer than 20 times
    batch_words=10000, # process corpus in 10,000-word chunks
    epochs=5,          # pass over the full corpus 5 times
    window=5           # consider 5 words on each side
)

model.save('model/wiki.model')
```

- `Text8Corpus` reads a whitespace-separated text file as a stream of sentences (lists of tokens).
- `vector_size=200` sets the dimension of each word vector.
- `min_count=20` filters out rare words to avoid learning noisy embeddings.
- `window=5` means the model predicts a target word from the 5 words before and 5 words after.
- `epochs=5` is the number of full passes over the corpus. Training logs show how many unique words were retained and how much of the corpus remains after `min_count` filtering.

## 2. Looking up a word vector

```python
print(model.wv['動物'])
len(model.wv['動物'])
```

- `model.wv` is the "word vectors" keyed object.
- `model.wv['動物']` returns the 200-dimensional NumPy vector for the word.
- If the word is not in the vocabulary, you get a `KeyError` — usually because it failed the `min_count` threshold.

## 3. Using word vectors

### 3.1 Finding similar words

```python
sample = '動物'
results = model.wv.most_similar(positive=[sample])
for result in results:
    print(result)
```

- `most_similar(positive=[word])` returns the words whose vectors are closest to the given word's vector.
- Each result is a `(word, cosine_similarity)` tuple.
- The returned words should be semantically or topically related to the query.

### 3.2 Vector arithmetic

```python
sample1 = '国会'
sample2 = '日本'
sample3 = '米国'

results = model.wv.most_similar(
    positive=[sample3, sample1],  # add these vectors
    negative=[sample2]            # subtract this vector
)
for result in results:
    print(result)
```

- `positive=[sample3, sample1]` computes `sample3 + sample1`.
- `negative=[sample2]` subtracts `sample2`.
- The query asks: "Which word is to 米国 as 国会 is to 日本?" — i.e. the analogue of Japan's parliament in the US context.
- Other commented examples include `生育 - 植物 + 動物` and `ロンドン - イギリス + 日本`.

### 3.3 Visualising word vectors with PCA

```python
from sklearn.decomposition import PCA

words = {
    '動物':'b','種':'b','細胞':'b','生物':'b','卵':'b','寄生':'b','飼育':'b','虫':'b','家畜':'b','犬':'b',
    '作品':'g','写真':'g','アニメ':'g','カメラ':'g','音楽':'g','テレビ':'g','スタジオ':'g','漫画':'g','表現':'g','賞':'g',
    '経済':'r','企業':'r','社会':'r','資本':'r','会社':'r','生産':'r','労働':'r','産業':'r','市場':'r','価格':'r',
    '法':'c','憲法':'c','国家':'c','監査':'c','権利':'c','解釈':'c','法律':'c','行政':'c','教会':'c','宗教':'c',
    '植物':'m','葉':'m','栽培':'m','品種':'m','茎':'m','種子':'m','枝':'m','根':'m','胞子':'m','果実':'m',
    '軍事':'y','選挙':'y','国民':'y','議会':'y','主権':'y','地方':'y','外交':'y','行政':'y','権力':'y','期間':'y'
}

data = [model.wv[word] for word in words.keys()]

pca = PCA(n_components=2)
pca.fit(data)
data_pca = pca.transform(data)
```

- 200-dimensional vectors cannot be plotted directly, so PCA reduces them to 2D.
- Each word is assigned a colour per category so you can see whether words from the same topic cluster together.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams["font.size"] = 18
plt.figure(figsize=(12, 9))

for i, word in enumerate(words):
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker='x', color=words[word])
    plt.annotate(word, (data_pca[i][0], data_pca[i][1]))

plt.grid()
plt.savefig('fig/Word2vec1.png')
```

- `japanize_matplotlib` handles Japanese font rendering.
- Words from the same category often form visible clusters, confirming that Word2vec captures topical similarity.

## 4. Hierarchical clustering

```python
%matplotlib inline
from scipy.cluster.hierarchy import linkage, dendrogram

plt.rcParams["font.family"] = "IPAexGothic"
plt.rcParams["font.size"] = 18

labels = list(words.keys())
l = linkage(data, method="ward")
plt.figure(figsize=(18, 10))
dendrogram(l, labels=labels, leaf_font_size=18)
plt.savefig('fig/Word2vec2.png')
```

- `linkage(data, method="ward")` builds a hierarchical cluster tree using Ward's method (minimises within-cluster variance).
- `dendrogram` draws the tree. Words that are close in vector space are joined lower in the tree.
- The clustering uses only vector distances; the category colours are not supplied to the algorithm. If the dendrogram mostly groups same-category words together, it confirms the embeddings encode category information.

## Key takeaways

- **Word2vec** learns dense word vectors from co-occurrence patterns in raw text.
- The trained vectors live in a space where similar words are close and analogies can sometimes be solved by vector arithmetic.
- `gensim.models.Word2Vec` parameters to remember: `vector_size` (dimension), `window` (context width), `min_count` (rare-word cutoff), and `epochs` (training passes).
- Word vectors can be reduced to 2D with PCA for visual inspection, or clustered with hierarchical methods to discover natural groupings.
- Because the corpus is small and domain-limited, some analogies work better than others; larger corpora produce more robust embeddings.
