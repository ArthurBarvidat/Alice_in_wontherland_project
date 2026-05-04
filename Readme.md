# bookworm
A lightweight NLP engine that creates **book cards** from Project Gutenberg books.
Built for publishers and editors who need to quickly understand a book without reading it entirely — turning long texts into structured summaries, topics, entities and more.

## Installation
pip install pandas nltk requests

## Usage

### Download a book
python bookworm.py --download 11
Downloads a book from the internet using its ID number and saves it on your computer. You need to do this first before using any other command.

### Get book info
python bookworm.py --infos 11
Shows basic information about a book like its title, who wrote it and what category it belongs to.

### Lexical diversity
python bookworm.py --lexdiv 11
Counts and analyzes the words used in the book — how many words there are in total, how many are unique, and how varied the author's vocabulary is.

### Topic modeling
python bookworm.py --topics 11
Reads through the book and finds the most important words in each section. Gives you a rough idea of what each part of the book is about.

### Named entity recognition
python bookworm.py --entities 11
Scans the book and pulls out the names of characters and places. Great for quickly knowing who is in the story and where it takes place.

### Summarize
python bookworm.py --summarize 11
Reads the whole book and writes a short summary of a few sentences. Perfect if you just want to know what the book is about without reading it.

### Similar books
python bookworm.py --similar 11
Compares the book with all the others in the collection and tells you which 5 books are the most similar. You need to have downloaded other books first for this to work.

### Full book card
python bookworm.py --card 11
Runs all the commands above at once and puts everything together in one big result. This is the main feature of the tool.

## Cache
Every time you run a command, the result is saved in a `cache/` folder. If you run the same command again, it will load the saved result instead of doing all the work again — which saves a lot of time.

## Example
python bookworm.py --download 11
python bookworm.py --download 12
python bookworm.py --download 16
python bookworm.py --card 11