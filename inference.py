import pandas as pd
import re
from natasha import (
    Segmenter, MorphVocab,
    NewsEmbedding, NewsNERTagger,
    NewsMorphTagger, Doc
)
from difflib import get_close_matches
from city_data import cities, city_mapping

# Инициализация Natasha компонентов
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def normalize_city_name(text):
    norm = text.lower().strip()
    if norm in city_mapping:
        return city_mapping[norm]
    return norm.capitalize()

def fuzzy_match_city(word):
    candidates = list(cities) + list(city_mapping.keys())
    match = get_close_matches(word, candidates, n=1, cutoff=0.8)
    if match:
        candidate = match[0]
        return city_mapping.get(candidate.lower(), candidate)
    return None

def extract_city(text):
    text_clean = text.lower().replace('.', '').replace('-', ' ')
    
    # 1. Поиск по city_mapping напрямую
    for alias, real in city_mapping.items():
        if re.search(rf'\b{alias}\b', text_clean):
            return real

    # 2. Морфология через Natasha
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)

    for span in doc.spans:
        if span.type == 'LOC':
            span.normalize(morph_vocab)
            city = span.normal
            if city in cities:
                return city
            fuzzy = fuzzy_match_city(city.lower())
            if fuzzy:
                return fuzzy

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma
        if lemma in cities:
            return lemma
        if lemma.lower() in city_mapping:
            return city_mapping[lemma.lower()]
        fuzzy = fuzzy_match_city(lemma.lower())
        if fuzzy:
            return fuzzy

    return None

def main():
    # Получение пути к файлу от пользователя
    file = input("Введите путь к файлу: ")
    
    # Чтение данных
    data = pd.read_csv(file, index_col=0)
    
    # Извлечение городов
    data['city'] = data['message'].apply(extract_city)
    
    # Сохранение результатов
    data.to_csv(file)
    print("Результаты сохранены в файл")

if __name__ == "__main__":
    main() 