# TextProcessing2.ipynb — Sub-word Tokenisation for LLMs

Notebook 1 split Japanese text into linguistically motivated morphemes using a hand-curated dictionary. Modern **Large Language Models (LLMs)** use a different strategy: they decide word boundaries **statistically**, splitting rare words into small pieces and keeping frequent words whole. The resulting units are called **subwords**, and the procedure is called **tokenising** (not "word segmentation") because the units aren't linguistic words.

The notebook demonstrates this using **BERT's WordPiece tokeniser**, distributed by Hugging Face `transformers`.

## Setup — install the libraries

```python
!pip install transformers fugashi unidic-lite ipadic
```

- `transformers` — Hugging Face's library; provides pre-trained tokenisers and models, including BERT.
- `fugashi` — another pure-Python wrapper around MeCab (similar in role to Janome from notebook 1).
- `unidic-lite` and `ipadic` — Japanese dictionary packages that fugashi can use.

## 1. Tokenising with fugashi first

```python
from fugashi import Tagger

tagger = Tagger()
line = '東京大学は欧米諸国の諸制度に倣った、日本国内で初の近代的な大学として設立された。'
for word in tagger.parseToNodeList(line):
    print(word, word.feature.lemma, word.pos, sep='\t')
```

- `Tagger()` builds a MeCab analyser (same engine family as Janome, different API).
- `parseToNodeList(text)` returns a list of nodes; each `word` exposes `.surface`, `.feature.lemma` (dictionary form), and `.pos` (part-of-speech tag).
- This step is included for comparison: fugashi's segmentation is **dictionary-based and fixed** — every model that uses it gets the same boundaries.

## 2. Loading the BERT Japanese tokeniser

```python
from transformers import BertJapaneseTokenizer
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
```

- `from_pretrained(model_name)` downloads the tokeniser config from the Hugging Face Hub and caches it locally.
- `cl-tohoku/bert-base-japanese-whole-word-masking` is the **Tohoku University** Japanese BERT, trained on Japanese Wikipedia. "Whole-word masking" means the masking strategy used during pretraining always masks every sub-piece of a word together — a refinement of plain BERT.
- `BertJapaneseTokenizer` is a wrapper that combines:
  1. A **MeCab-based** pre-tokeniser (using fugashi under the hood) for word boundaries.
  2. A **WordPiece** sub-word model trained over the resulting word stream.

## 3. Tokenising a sentence

```python
line = '東京大学は欧米諸国の諸制度に倣った、日本国内で初の近代的な大学として設立された。'
token = tokenizer.tokenize(line)
token
```

- `tokenize(text)` returns the list of sub-word strings. For the input above, the output is roughly:

  ```
  ['東京', '大学', 'は', '欧米', '諸', '国', 'の', '諸', '制度', 'に', '倣っ', 'た', '、', '日本', '国内', 'で', '初', 'の', '近代', '的', 'な', '大学', 'として', '設立', 'され', 'た', '。']
  ```

- Note how **common words** like 「大学」, 「日本」, 「国内」 appear as single tokens, while **rare compounds** like 「東京大学」 get split into 「東京」 + 「大学」. This is the "common = whole, rare = split" rule in action.

### From tokens to IDs

```python
input_ids = tokenizer.encode(line)
input_ids
```

- `encode(text)` runs the full pipeline: tokenise → add special tokens → convert to integer IDs.
- Each unique sub-word has a fixed integer ID — these IDs are what the neural network actually consumes.
- The output begins with `2` (the `[CLS]` "classification" token that BERT prepends to every input) and ends with `3` (the `[SEP]` separator).
- Calling `encode` again on a different sentence shows the same IDs for the same words:

  ```python
  tokenizer.tokenize('東京大学は大学だ。')
  # ['東京', '大学', 'は', '大学', 'だ', '。']
  tokenizer.encode('東京大学は大学だ。')
  # [2, ..., 3]
  ```

  Within a single model, identical words always map to identical IDs — the mapping is deterministic.

## 4. Why LLMs use sub-words, and why English is cheaper than Japanese

The notebook's final markdown cell explains the practical consequence of sub-word tokenisation:

- **English** is internationally frequent — about **75 words per 100 tokens**. Each English word tends to map to ~1.3 tokens on average.
- **Japanese** characters are individually rarer, so:
  - 「の」 — frequent, 1 character → **1 token**
  - 「日」 — rarer, 1 character → **2 tokens**
  - 「音」 — even rarer, 1 character → **3 tokens**

- ChatGPT's context window is **4 097 tokens** (GPT-3). The same window holds more English than Japanese because Japanese text expands into more tokens per character.

For stateful (conversation-style) LLMs, this directly translates to **how much of the prior conversation the model can "remember"** — the English conversation remembers more.

The OpenAI website has a [live tokeniser demo](https://platform.openai.com/tokenizer) where you can paste Japanese or English text and watch it split.

## Why "tokenise" and not "segment"?

The notebook's introduction makes an important terminological point:

- Janome / MeCab split text along **linguistically motivated** boundaries → "形態素解析" / "分かち書き".
- BERT-style tokenisers split text along **statistically optimal** boundaries driven by frequency → "トークナイジング".

The pieces aren't linguistic units — calling them "tokens" instead of "words" makes it clear that the boundaries are machine-learned, not language-theoretic.

## Key takeaways

- **Sub-word tokenisation** adapts word length to word frequency: common words stay whole, rare words split into pieces.
- **WordPiece** (BERT) is one sub-word algorithm; BPE (GPT-family) and SentencePiece (LLaMA, T5) are others.
- The same model always maps the same sub-word to the same integer ID — that's how LLMs treat text as numerical input.
- **Japanese is token-inefficient compared to English**: each kanji character can take 1–3 tokens, while each English word is usually 1 token. This affects how much text an LLM can process in a single context window.
- `transformers.BertJapaneseTokenizer` is the standard way to load a Japanese BERT tokeniser; under the hood it uses fugashi (MeCab) for word pre-segmentation and WordPiece for the sub-word split.
