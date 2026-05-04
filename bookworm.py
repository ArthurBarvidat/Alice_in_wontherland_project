import pandas as pd
import argparse
import nltk
import requests
import json
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

_lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# book collection available for comparison
BOOKS = {
    11: "Alice's Adventures in Wonderland",
    12: "Through the Looking-Glass",
    16: "Peter Pan",
    55: "The Wonderful Wizard of Oz",
    113: "The Secret Garden",
    120: "Treasure Island",
    236: "The Jungle Book",
    108: "The Return of Sherlock Holmes",
    834: "The Memoirs of Sherlock Holmes",
    863: "The Mysterious Affair at Styles",
    1661: "The Adventures of Sherlock Holmes",
    61262: "Poirot Investigates",
    69087: "The murder of Roger Ackroyd",
    70114: "The Big Four",
    35: "The Time Machine",
    36: "The War of the Worlds",
    84: "Frankenstein; Or, The Modern Prometheus",
    159: "The island of Doctor Moreau",
    164: "Twenty Thousand Leagues under the Sea",
    345: "Dracula",
    68283: "The call of Cthulhu",
}

# define available command line options
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--infos", type=int)
group.add_argument("--lexdiv", type=int)
group.add_argument("--topics", type=int)
group.add_argument("--entities", type=int)
group.add_argument("--summarize", type=int)
group.add_argument("--similar", type=int)
group.add_argument("--download", type=int)
group.add_argument("--card", type=int)
args = parser.parse_args()


def load_cache(key):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# fetch book information from the catalog
def get_info(id_book):
    data = pd.read_csv("pg_catalog.csv")
    ligne = data[data["Text#"] == id_book]
    if ligne.empty:
        return {"error": "book not found"}
    return {
        "id": str(id_book),
        "authors": str(ligne["Authors"].values[0]),
        "bookshelves": str(ligne["Bookshelves"].values[0])}


if args.infos:
    print(get_info(args.infos))


# download the book from Project Gutenberg
if args.download:
    url = f"https://www.gutenberg.org/files/{args.download}/{args.download}-0.txt"
    requete = requests.get(url)
    if requete.status_code == 200:
        with open(f"{args.download}.txt", "w", encoding="utf-8") as fichier:
            fichier.write(requete.text)


# compute lexical diversity metrics of the book
def get_lexdiv(id_book):
    cached = load_cache(f"lexdiv_{id_book}")
    if cached:
        return cached

    texte = open(f"{id_book}.txt", encoding="utf-8").read().lower()
    tokens = nltk.word_tokenize(texte)
    tok = len(tokens)
    typ = len(set(tokens))
    freq = {}
    for mot in tokens:
        if mot in freq:
            freq[mot] = freq[mot] + 1
        else:
            freq[mot] = 1
    hap = 0
    for mot in freq:
        if freq[mot] == 1:
            hap = hap + 1
    if tok != 0:
        ttr = typ / tok
        mwl = sum(len(m) for m in tokens) / tok
    else:
        ttr = 0
        mwl = 0
    if typ != 0:
        mwf = tok / typ
    else:
        mwf = 0

    result = {"tok": tok, "typ": typ, "hap": hap, "ttr": ttr, "mwl": mwl, "mwf": mwf}
    save_cache(f"lexdiv_{id_book}", result)
    return result


if args.lexdiv:
    print(get_lexdiv(args.lexdiv))


# extract main words from each section of the book
def get_topics(id_book):
    cached = load_cache(f"topics_{id_book}")
    if cached:
        return {int(k): v for k, v in cached.items()}

    texte = open(f"{id_book}.txt", encoding="utf-8").read().lower()
    stop_words = stopwords.words("english")

    # split the text into 4 sections
    paragraphes = [p.strip() for p in texte.split("\n\n") if len(p.strip()) > 100]
    taille = max(1, len(paragraphes) // 4)
    sections = [
        " ".join(paragraphes[i * taille:(i + 1) * taille])
        for i in range(4)
    ]

    result = {}
    for i in range(4):
        mots = nltk.word_tokenize(sections[i])
        freq = {}
        for mot in mots:
            if mot.isalpha() and mot not in stop_words:
                freq[mot] = freq.get(mot, 0) + 1
        result[i + 1] = sorted(freq, key=freq.get, reverse=True)[:10]

    save_cache(f"topics_{id_book}", result)
    return result


if args.topics:
    print(get_topics(args.topics))


# identify characters and locations in the book
def get_entities(id_book):
    cached = load_cache(f"entities_{id_book}")
    if cached:
        return cached

    texte = open(f"{id_book}.txt", encoding="utf-8").read()
    mots = nltk.word_tokenize(texte)
    etiquettes = nltk.pos_tag(mots)
    arbre = nltk.ne_chunk(etiquettes)
    characters = []
    locations = []
    for element in arbre:
        if hasattr(element, "label"):
            nom = " ".join([mot[0] for mot in element])
            if element.label() == "PERSON":
                characters.append(nom)
            elif element.label() in ["GPE", "LOCATION"]:
                locations.append(nom)

    result = {"characters": characters[:10], "locations": locations[:10]}
    save_cache(f"entities_{id_book}", result)
    return result


if args.entities:
    print(get_entities(args.entities))


# summarize the book in a few sentences
def get_summary(id_book):
    cached = load_cache(f"summary_{id_book}")
    if cached:
        return cached

    texte = open(f"{id_book}.txt", encoding="utf-8").read()
    if "*** START OF" in texte:
        texte = texte.split("*** START OF")[1].split("***")[1]
    sentences = nltk.sent_tokenize(texte)

    stop_words = set(stopwords.words("english"))
    freq = {}
    for mot in nltk.word_tokenize(texte.lower()):
        if mot.isalpha() and mot not in stop_words:
            freq[mot] = freq.get(mot, 0) + 1

    scores = {}
    for sentence in sentences:
        score = 0
        for mot in nltk.word_tokenize(sentence.lower()):
            if mot in freq:
                score = score + freq[mot]
        scores[sentence] = score

    # keep the top 5 in their order of appearance
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    top5 = [s for s in sentences if s in top5]
    result = " ".join(top5)

    save_cache(f"summary_{id_book}", result)
    return result


if args.summarize:
    print(get_summary(args.summarize))


# helpers for similar
def _tokenize_words(text):
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 1]


def _lemmatize_tokens(tokens):
    return [_lemmatizer.lemmatize(t) for t in tokens]


def _get_book_text(bid):
    path = f"{bid}.txt"
    if not os.path.exists(path):
        return ""
    return open(path, encoding="utf-8", errors="replace").read()


# find the most similar books using TF-IDF + cosine similarity
def get_similar(id_book):
    cached = load_cache(f"similar_{id_book}")
    if cached:
        return cached

    if not os.path.exists(f"{id_book}.txt"):
        return {"error": f"book {id_book} not downloaded"}

    all_ids = list(BOOKS.keys())
    all_texts = []

    for bid in all_ids:
        text = _get_book_text(bid)
        tokens = _lemmatize_tokens(_tokenize_words(text)) if text else []
        all_texts.append(" ".join(tokens))

    vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(all_texts)

    if id_book in all_ids:
        target_idx = all_ids.index(id_book)
        target_vec = tfidf[target_idx]
        sims = cosine_similarity(target_vec, tfidf)[0]
        ranked = sorted(
            [i for i in range(len(all_ids)) if i != target_idx],
            key=lambda i: sims[i],
            reverse=True,
        )
    else:
        target_text = _get_book_text(id_book)
        target_tokens = _lemmatize_tokens(_tokenize_words(target_text))
        target_vec = vectorizer.transform([" ".join(target_tokens)])
        sims = cosine_similarity(target_vec, tfidf)[0]
        ranked = sorted(range(len(all_ids)), key=lambda i: sims[i], reverse=True)

    result = [BOOKS[all_ids[i]] for i in ranked[:5]]
    save_cache(f"similar_{id_book}", result)
    return result


if args.similar:
    print(get_similar(args.similar))


# compile all information into a book card
def get_card(id_book):
    return {
        "info": get_info(id_book),
        "lexdiv": get_lexdiv(id_book),
        "topics": get_topics(id_book),
        "entities": get_entities(id_book),
        "summary": get_summary(id_book),
        "similar": get_similar(id_book)}


if args.card:
    print(get_card(args.card))