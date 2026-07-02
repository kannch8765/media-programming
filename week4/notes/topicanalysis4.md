# TopicAnalysis4.ipynb — Solving NLP Tasks with Pre-trained BERT

This notebook shows how to use **pre-trained transformer models** from Hugging Face for a variety of NLP tasks. It starts with Japanese BERT for fill-mask, then switches to English models for sentiment analysis, question answering, text generation, named entity recognition, summarisation, and translation.

## 1. Installing the libraries

```python
!pip install transformers
!pip install fugashi
!pip install unidic-lite
!pip install ipadic
!pip install sentencepiece==0.1.94
!pip install sacremoses
```

- `transformers` is the Hugging Face library that provides model architectures and pre-trained weights.
- `fugashi`, `unidic-lite`, `ipadic` are required by the Japanese BERT tokenizer.
- `sentencepiece` and `sacremoses` are tokenisation backends used by some multilingual models.

```python
from transformers import (
    BertTokenizer,
    BertJapaneseTokenizer,
    BertForSequenceClassification,
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)
```

## 2. Japanese fill-mask with BERT

```python
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
unmasker = pipeline('fill-mask', model=model_name, tokenizer=tokenizer)
```

- `cl-tohoku/bert-base-japanese-whole-word-masking` is the Tohoku University Japanese BERT.
- `BertJapaneseTokenizer.from_pretrained(...)` downloads the matching tokenizer.
- `pipeline('fill-mask', ...)` wraps the model+tokenizer into a task-specific pipeline.

```python
unmasker("東京駅から[MASK]駅に向かいます。")
```

- `[MASK]` is the placeholder BERT replaces.
- The pipeline returns the top candidate tokens with confidence scores.
- Because the model was trained mainly on Wikipedia, it works best on formal, encyclopedic Japanese.

## 3. English NLP tasks with `pipeline`

### 3.1 Fill-mask (English)

```python
unmasker = pipeline('fill-mask', model='distilroberta-base')
unmasker("He bought a gallon <mask> milk.")
```

- `distilroberta-base` is a smaller English model; `<mask>` is its mask token.

### 3.2 Sentiment analysis

```python
classifier = pipeline(
    "sentiment-analysis",
    model='distilbert-base-uncased-finetuned-sst-2-english'
)
print(classifier("I hate you")[0])
print(classifier("I love you")[0])
```

- Returns `POSITIVE` or `NEGATIVE` plus a confidence score.
- This exact model is reused in `mp_ex4.ipynb` to find the most positive sentence.

### 3.3 Question answering

```python
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

context = r"""UTokyo has ten faculties, 15 graduate schools and enrolls about 30,000 students, about 4,200 of whom are international students.
Its five campuses are in Hongō, Komaba, Kashiwa, Shirokane and Nakano.
It is considered to be the most selective and prestigious university in Japan."""

questions = [
    "How many international students are there at the UTokyo?",
    "Where is the campus located?",
    "What kind of university is the University of Tokyo known for?"
]

def answer_question(question, context, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)

    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1

    input_ids = inputs['input_ids'].tolist()[0]
    answer_tokens = input_ids[answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {
        'answer': answer,
        'start': answer_start.item(),
        'end': answer_end.item(),
        'score': (outputs.start_logits[0, answer_start].item() +
                  outputs.end_logits[0, answer_end - 1].item()) / 2
    }

for question in questions:
    print(answer_question(question, context, tokenizer, model))
```

- Extractive QA finds the span in `context` that best answers `question`.
- The tokenizer concatenates question and context; the model outputs start/end logits for the answer span.
- `max_length=512` and `truncation=True` prevent inputs that are too long for the model.

### 3.4 Text generation

```python
text_generator = pipeline("text-generation", model='gpt2', max_length=50, pad_token_id=50256)
text_generator("I was so surprised that the university of Tokyo was")[0]['generated_text']
```

- `gpt2` is an autoregressive language model: it generates continuations token by token.
- Output is non-deterministic because sampling is random.
- `pad_token_id=50256` suppresses a warning by telling the model which token to use for padding.

### 3.5 Named entity recognition (NER)

```python
ner_pipe = pipeline("ner", model='dbmdz/bert-large-cased-finetuned-conll03-english')
sequence = """The University of Tokyo is a public research university located in Bunkyō, Tokyo, Japan. The president of the university is Teruo Fujii."""
result = ner_pipe(sequence)
for word in result:
    print(word)
```

- NER labels each token as a person (PER), organisation (ORG), location (LOC), etc.
- Useful as a pre-processing step for relation extraction or information retrieval.

### 3.6 Summarisation

```python
tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')

inputs = tokenizer(ARTICLE, max_length=1024, truncation=True, return_tensors="pt")
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=130,
    min_length=30,
    early_stopping=True
)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary_text)
```

- Seq2Seq models generate a shorter summary from a long input article.
- `num_beams=4` uses beam search to balance quality and speed.
- `max_length` and `min_length` constrain the output length.

### 3.7 Translation

#### Japanese → English

```python
tokenizer_ja_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
model_ja_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

japanese_text = "東京大学はいろいろな学問が学べて楽しい。"
input_ids = tokenizer_ja_en.encode(japanese_text, return_tensors="pt")
translated_ids = model_ja_en.generate(input_ids)
english_text = tokenizer_ja_en.decode(translated_ids[0], skip_special_tokens=True)
print(english_text)
```

- `Helsinki-NLP/opus-mt-ja-en` is a pre-trained Japanese-to-English translation model.
- Translation is also a sequence-to-sequence task.

#### English → German / French with T5

```python
tokenizer_t5 = AutoTokenizer.from_pretrained('t5-base')
model_t5 = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

input_text_de = "translate English to German: " + english_text
input_ids_de = tokenizer_t5.encode(input_text_de, return_tensors="pt")
translated_ids_de = model_t5.generate(input_ids_de, max_length=512)
german_text = tokenizer_t5.decode(translated_ids_de[0], skip_special_tokens=True)
print('ドイツ語:', german_text)

input_text_fr = "translate English to French: " + english_text
input_ids_fr = tokenizer_t5.encode(input_text_fr, return_tensors="pt")
translated_ids_fr = model_t5.generate(input_ids_fr, max_length=512)
french_text = tokenizer_t5.decode(translated_ids_fr[0], skip_special_tokens=True)
print('フランス語:', french_text)
```

- T5 is a unified text-to-text model; translation is invoked by prepending a task prefix like `translate English to German:`.
- English is a common pivot language for low-resource language pairs.

## Key takeaways

- **BERT** and related transformer models are pre-trained on large corpora and then fine-tuned or used directly for many NLP tasks.
- Hugging Face `pipeline` is the quickest way to run a task: fill-mask, sentiment analysis, QA, NER, text generation, summarisation, and translation.
- For direct model use, the pattern is: load `AutoTokenizer` and `AutoModelFor<Task>`, tokenise inputs, run the model, and decode outputs.
- Japanese BERT needs a Japanese-compatible tokenizer (`BertJapaneseTokenizer`) and supporting dictionary packages.
- Translation and summarisation are seq2seq tasks; QA and NER are encoder-only or encoder-question tasks; text generation is autoregressive.
- Pre-trained models are powerful, but their knowledge comes from their training data — they struggle with slang, recent events, and out-of-domain text.
