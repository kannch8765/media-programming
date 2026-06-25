# TextProcessing1.ipynb — Morphological Analysis with Janome

This notebook is the first in the "Natural Language Processing" series. It introduces **morphological analysis** (形態素解析) — splitting Japanese text into its smallest meaningful units (morphemes) — using the pure-Python library **Janome**.

The example sentence used throughout is:

> 東京大学は欧米諸国の諸制度に倣った、日本国内で初の近代的な大学として設立された。

(from the Japanese Wikipedia article on "東京大学").

The notebook also demonstrates user-defined dictionaries, noun-frequency histograms, and word-cloud visualisation over Aozora Bunko novels.

## Setup cell — download the data

```python
import urllib.request
import os
import zipfile

os.makedirs('fig', exist_ok=True)
os.makedirs('text', exist_ok=True)
urllib.request.urlretrieve(
    'https://park.itc.u-tokyo.ac.jp/yamakata-lab/lecture/mediaproc/mediaproc3/mediaproc3-1.zip',
    './text/mediaproc3-1.zip'
)
with zipfile.ZipFile('text/mediaproc3-1.zip') as zip:
    zip.extractall('text/')

# On Colab only:
if 'google.colab' in sys.modules:
    !apt-get -y install fonts-ipafont-gothic
    !rm /root/.cache/matplotlib/*.json
```

- `urllib.request.urlretrieve(url, path)` downloads a remote file directly to disk.
- `zipfile.ZipFile(path).extractall(target)` unzips the archive into `target`.
- `os.makedirs(..., exist_ok=True)` creates the directory if it doesn't exist; `exist_ok=True` avoids an error when it already does.
- The Colab-only block installs **IPAfont Gothic** (a free Japanese font) and clears matplotlib's font cache so the freshly installed font gets picked up.
- A reminder markdown cell tells Colab users to **restart the runtime** after installation — the font change only takes effect for new matplotlib processes.

## Installing Janome

```python
!pip install janome
```

- Janome is a pure-Python morphological analyser with an embedded dictionary.
- No C dependencies, so it installs cleanly even on systems without a MeCab build.

## 1.1 Running the analyser

```python
from janome.tokenizer import Tokenizer

t = Tokenizer()  # loads the dictionary; expensive — call once per program

line = '東京大学は欧米諸国の諸制度に倣った、日本国内で初の近代的な大学として設立された。'

for token in t.tokenize(line):
    print(token)
```

- `Tokenizer()` loads the bundled IPAdic dictionary into memory and initialises the analysis engine. Because dictionary loading is slow, instantiate the `Tokenizer` **once** and reuse the object.
- `t.tokenize(line)` returns an iterator of `Token` objects. Each `Token`, when printed, shows the full analysis line:

  ```
  表層形, 品詞, 品詞細分類1, 品詞細分類2, 品詞細分類3, 活用型, 活用形, 原形, 読み, 発音
  ```

- A **morpheme** (形態素) is the smallest meaningful unit of language — smaller than what most people call a "word". For example:
  - 「不確実」 → 「不」 (prefix) + 「確実」 (noun)
  - 「おいしさ」 → 「おいし」 (adjective stem) + 「さ」 (suffix)

  These splits come from IPAdic's definitions, which are tuned for training-corpus convenience rather than strict linguistics, so results are not always linguistically "correct".

## 1.2 Accessing token attributes

```python
for token in t.tokenize(line):
    print(token.surface)   # 表層形 (surface form)
```

- The available attributes are:
  - `surface` — the visible characters (表層形)
  - `part_of_speech` — POS tag (品詞), as a comma-joined 4-tuple
  - `infl_type` — inflection type (活用型)
  - `infl_form` — inflection form (活用形)
  - `base_form` — dictionary / lemma form (基本形)
  - `reading` — katakana reading
  - `phonetic` — katakana pronunciation

## 1.3 Word segmentation (分かち書き)

```python
print('/'.join(t.tokenize(line, wakati=True)))
```

- `wakati=True` tells Janome to skip the full analysis and return only the surface forms as a list of strings.
- Inserting a delimiter (here `/`) between the tokens yields a **分かち書き** ("separated writing") form, which is the canonical preprocessing output for almost all Japanese NLP pipelines.

## 1.4 POS and Named Entities

```python
for token in t.tokenize(line):
    print(token.part_of_speech)
```

- `part_of_speech` returns the four-level POS tuple as a comma-joined string:

  ```
  品詞,品詞細分類1,品詞細分類2,品詞細分類3
  ```

- Levels 2 and 3 carry IREX **Named Entity (NE)** tags — 8 categories used in the 1998–1999 IREX information-retrieval contest:

  | Category | Japanese |
  | --- | --- |
  | ORGANIZATION | 組織名 |
  | PERSON       | 人名 |
  | LOCATION     | 地名 |
  | DATE         | 日付表現 |
  | TIME         | 時間表現 |
  | MONEY        | 金額表現 |
  | PERCENT      | 割合表現 |
  | ARTIFACT     | 固有物名 |

## 2. Adding a user-defined dictionary

Janome (like all statistical morphological analysers) chooses the most plausible segmentation from a **fixed dictionary**. Anything not in the dictionary cannot be recognised as a single morpheme — and a single unknown word often mis-segments the words around it too.

This is a real problem for newspaper-trained dictionaries when applied to SNS text, where slang, kaomoji, and brand-new proper nouns dominate.

The notebook demonstrates by analysing:

> ロスマリン酸摂取後の脳内ドーパミンがアルツハイマー病を予防する ポリフェノールの新たな作用機序

without a user dictionary, then with one.

### 2.1 Dictionary file format

Janome accepts MeCab-format CSV, one entry per line:

```
表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
```

- **Left/right context IDs** identify the POS state on each side; Janome auto-assigns them if you set them to `-1`.
- **Cost** is a non-negative integer: smaller values = more likely. Start at the same value as similar entries, then nudge down if the analyser refuses to pick the entry.
- The CSV may carry extra user-defined columns after `発音`; they are ignored by the analyser.

A useful reference is `right-id.def` / `left-id.def` from the unpacked IPAdic, which lists every POS state and its numeric ID. These IDs are the keys into `matrix.def`, the table of inter-POS connection costs that drives the Viterbi-style search for the best segmentation.

### 2.2 Creating the user dictionary

```python
with open('userdic.csv', 'w', encoding='utf-8') as f:
    f.write('ロスマリン,-1,-1,1000,名詞,固有名詞,一般,*,*,*,ロスマリン,ロスマリン,ロスマリン\n')
    f.write('ポリフェノール,-1,-1,1000,名詞,固有名詞,一般,*,*,*,ポリフェノール,ポリフェノール,ポリフェノール\n')
    f.write('機序,-1,-1,1000,名詞,普通名詞,*,*,*,*,機序,キジョ,キジョ\n')
```

- Three proper-noun / scientific-term entries are written in CSV form.
- Cost `1000` is a neutral starting point. Adjust if you find a word being over- or under-segmented.
- `encoding='utf-8'` is essential — Janome's default encoding (`udic_enc`) is also `utf-8`, so the file must match.

### 2.3 Loading and re-analysing

```python
from janome.tokenizer import Tokenizer
t = Tokenizer("userdic.csv", udic_enc="utf8")

line = 'ロスマリン酸摂取後の脳内ドーパミンがアルツハイマー病を予防する ポリフェノールの新たな作用機序'

for token in t.tokenize(line):
    print(token)
```

- The second argument to `Tokenizer(path, udic_enc=...)` activates the user dictionary. The same `Tokenizer` instance is rebound, so any prior analysis state is discarded.
- After loading, 「ロスマリン」, 「ポリフェノール」, and 「機序」 appear as single morphemes instead of being chopped into fragments.

## 3.1 Histogram of nouns in an Aozora Bunko novel

```python
import csv
import collections

noun = []
with open('text/dazai_shayou_morphem.csv', 'r', encoding='utf-8') as fi:  # 「斜陽」
    dataReader = csv.reader(fi)
    for row in dataReader:
        if row[1] == '名詞' and row[0] != 'つて' and row[0] != 'の' and row[0] != 'やう':
            noun.append(row[0])

count = collections.Counter(noun)
```

- Each `text/*_morphem.csv` is a pre-tokenised Aozora Bunko novel. The CSV rows are: `[表層形, 品詞, ...]`, so `row[0]` is the surface form and `row[1]` is the top-level POS.
- `collections.Counter(iterable)` builds a `dict` of `{item: frequency}`. Sorting by `.most_common()` gives you the ranking.
- The `row[0] != ...` exclusions filter out **old-kana orthography** (`つて`, `の`, `やう`) that IPAdic frequently mis-labels as nouns. The notebook author admits this is a hack — the proper fix is a user dictionary for old-kana forms.
- Other novels in the corpus (toggle by commenting/uncommenting):
  - `miyazwa_chuumon_morphem.csv` — 「注文の多い料理店」
  - `miyazawa_gingatetsudo_morphem.csv` — 「銀河鉄道の夜」
  - `miyazawa_GauchetheCellist_morphem.csv` — 「セロ弾きのゴーシュ」
  - `miyazawa_kazenomatasaburou_morphem.csv` — 「風の又三郎」
  - `dazai_ningenshikkaku_morphem.csv` — 「人間失格」
  - `dazai_RunMelos_morphem.csv` — 「走れメロス」

```python
%matplotlib inline
from operator import itemgetter
import matplotlib.pyplot as plt

sorted_count = sorted(count.items(), key=itemgetter(1), reverse=True)
label = [x[0] for x in sorted_count]
value = [x[1] for x in sorted_count]

plt.rcParams["font.family"] = "IPAGothic"   # Colab
# plt.rcParams['font.family'] = 'AppleGothic'   # Mac
# plt.rcParams["font.family"] = "Yu Mincho"      # Windows
# plt.rcParams["font.family"] = "IPAexGothic"    # if installed manually

plt.rcParams["font.size"] = 18
plt.figure(figsize=(12, 9))
plt.xticks(rotation=45, horizontalalignment='right')
plt.plot(label[:30], value[:30])
plt.savefig('fig/TextProcessing1-1.png')
```

- `itemgetter(1)` pulls the count out of each `(word, count)` tuple for sorting.
- `label[:30]` and `value[:30]` truncate to the top-30 most frequent nouns — long tails become unreadable.
- The font line is commented per OS — without a Japanese-capable font, kanji will render as `□`.

## 3.2 Word cloud

```python
%matplotlib inline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(
    background_color="white",
    font_path='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',  # Colab
    width=640, height=480
).generate_from_frequencies(count)

fig = plt.figure(figsize=(12, 8))
plt.imshow(wc)
plt.axis("off")
fig.tight_layout()
plt.savefig('fig/TextProcessing1-2.png')
```

- `WordCloud(...).generate_from_frequencies(count)` lays out each word at a font size proportional to its frequency, in a randomly-packed layout.
- `font_path` is mandatory for Japanese; point it at whatever Japanese font you installed.
- `plt.axis("off")` hides the axes — word clouds are decorative.
- `tight_layout()` trims padding so labels aren't clipped.

## Key takeaways

- A **morpheme** is smaller than a "word"; morphological analysis splits Japanese text into these smallest meaningful pieces using a dictionary + statistical model.
- **Janome** is a pure-Python analyser with an embedded IPAdic dictionary — easy to install, no MeCab build required.
- The `Token` object exposes surface form, POS, inflection, base form, reading, and pronunciation.
- **User dictionaries** are essential for proper nouns, scientific terms, and SNS-style text not covered by the default newspaper-trained dictionary.
- Noun-frequency analysis over tokenised novels is the canonical "first look" at a text — bar charts and word clouds both work.
- Always pick a Japanese-capable font (`IPAGothic` on Colab) before plotting kanji.
