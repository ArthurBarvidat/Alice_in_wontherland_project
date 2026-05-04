# Alice's Adventures in Wonderland — NLP Book Analysis Engine

A lightweight NLP engine that generates "book cards" from Project Gutenberg books, helping publishers and editors quickly understand a book without reading it entirely.

## What it does

The tool performs several NLP operations on books via a CLI script (`bookworm.py`):

| Option | Description |
|---|---|
| `--lexdiv <ID>` | Lexical diversity metrics (token count, type-token ratio, hapax legomena, etc.) |
| `--topics <ID>` | Topic modeling — extracts top 10 keywords per section |
| `--entities <ID>` | Named Entity Recognition — extracts characters and locations |
| `--summarize <ID>` | Summarizes the book in a few sentences |
| `--similar <ID>` | Returns 5 similar books sorted by decreasing similarity |
| `--card <ID>` | Full book card combining all the above |

Results are cached to avoid recomputing expensive operations.

## Usage

```bash
python bookworm.py --lexdiv 11
python bookworm.py --topics 11
python bookworm.py --entities 11
python bookworm.py --summarize 11
python bookworm.py --similar 11
python bookworm.py --card 11
```

The ID corresponds to the Project Gutenberg book ID (e.g. `11` = Alice's Adventures in Wonderland).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python bookworm.py --card 11
```

## Tech stack

- Python
- NLTK, spaCy, scikit-learn
- No heavy models (no LLMs, no large transformers)

## Project context

Built as part of an Epitech NLP project. The goal was to understand and justify NLP pipeline choices (extractive vs abstractive summarization, LDA vs LSA for topic modeling, cosine similarity for book recommendations).
