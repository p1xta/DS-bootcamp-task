import re
from functools import lru_cache

from bs4 import BeautifulSoup
import pymorphy2


_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

_morph = pymorphy2.MorphAnalyzer()


def clean_html(html: str) -> str:
    """
    Убирает HTML-разметку, скрипты/стили, схлопывает пробелы.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_indexed_text(title: str, body_html: str, title_repeat: int = 3) -> str:
    """
    Формирует текст статьи для индексации.
    """
    clean_body = clean_html(body_html)
    title = (title or "").strip()
    repeated_title = (title + " ") * max(title_repeat, 1)
    return f"{repeated_title.strip()} {clean_body}".strip()


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    if _morph is None:
        return token
    return _morph.parse(token)[0].normal_form


def tokenize(text: str, lemmatize: bool = True) -> list[str]:
    """
    Токенизация и лемматизация для BM25.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if lemmatize and _morph is not None:
        return [_lemma(t) for t in tokens]
    return tokens
