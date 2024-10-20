import re
import string
from pathlib import Path
from random import randbytes
import contractions
import datasets
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .profanity_patterns import RE_PATTERNS

nltk.download('english')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


def prepare(raw_data: Path) -> datasets.Dataset:
    dataset = pd.read_excel(raw_data)
    print(len(dataset))
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()
    print(len(dataset))
    dataset["message"] = dataset["message"].apply(lambda x: normalize(preprocess(x)))
    dataset["message"] = dataset["message"].apply(lambda x: [w for w in x if len(w) > 0])
    dataset["message"] = dataset["message"][dataset["message"].apply(len) > 0]
    dataset = dataset.dropna().reset_index(drop=True)
    dataset["message"] = dataset["message"].apply(lambda x: ' '.join(x))
    dataset = dataset.drop_duplicates(subset=["message"])
    return datasets.Dataset.from_pandas(dataset)


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))


def replace_contractions(text):
    return contractions.fix(text)


def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


def remove_repeated_symbols(text):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1", text)


def cap2word(text):
    delimiter = str(randbytes(100))

    def apply_delimiter(match):
        match = str(match.group())
        match = match[0] + match[-1]
        return match.lower() + delimiter

    spaces = re.sub(r'[A-Z] [A-Z]', apply_delimiter, text)
    return spaces.replace(delimiter + ' ', '', spaces.count(delimiter) - 1).replace(delimiter, '')


def remove_img(sample):
    return re.sub(r"\[img\]\S+\[/img\]", ' ', sample)


def strip_ip(s):
    try:
        return s.replace(re.compile('(([2][5][0-5]\.)|([2][0-4][0-9]\.)|([0-1]?[0-9]?[0-9]\.)){3}'
                                    + '(([2][5][0-5])|([2][0-4][0-9])|([0-1]?[0-9]?[0-9]))').search(s).group(), ' ')
    except:
        return s


def replace_URL(sample):
    return re.sub(r"(https://)([.\w]+)(/\S+)", r"\2", sample)


def remove_non_ascii(words):
    return [word.encode('ascii', 'ignore').decode('ascii') for word in words]


def remove_punctuation(text):
    return re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]').sub(" ", text)


def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='n')
        lemma = lemmatizer.lemmatize(lemma, pos='v')
        lemma = lemmatizer.lemmatize(lemma, pos='a')
        lemma = lemmatizer.lemmatize(lemma, pos='r')
        lemma = lemmatizer.lemmatize(lemma, pos='s')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words


def preprocess(sample):
    sample = remove_img(sample)
    sample = replace_URL(sample)
    sample = strip_ip(sample)
    sample = cap2word(sample)
    sample = sample.lower()
    sample = remove_repeated_symbols(sample)

    for target, patterns in RE_PATTERNS.items():
        for pat in patterns:
            sample = re.sub(pat, target, sample)
    sample = re.sub(r"[^a-z' ]", ' ', sample)

    sample = replace_contractions(sample)
    sample = remove_punctuation(sample)

    words = nltk.word_tokenize(sample)
    words = lemmatize_verbs(words)
    return normalize(words)
