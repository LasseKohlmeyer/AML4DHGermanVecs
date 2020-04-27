from typing import List
import spacy
from gensim.models import Word2Vec, Phrases


def tokens_from_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = f.read()

    return data.split("\n")


def preprocess(tokens: List[str], lemmatize: bool = True, lower: bool = True,
               pos_filter: list = None, remove_stopwords: bool = True,
               remove_punctuation: bool = True, lan_model = None) -> List[str]:
    def token_representation(token):
        representation = str(token.lemma_) if lemmatize else str(token)
        if lower:
            representation = representation.lower()
        return representation

    nlp = spacy.load("de_core_news_sm") if lan_model is None else lan_model
    nlp.Defaults.stop_words |= {"der", "die", "das", "Der", "Die", "Das"}

    for word in nlp.Defaults.stop_words:
        lex = nlp.vocab[word]
        lex.is_stop = True
    preprocessed_tokens = []

    if pos_filter is None:
        for doc in nlp.pipe(tokens, disable=['parser', 'ner', 'tagger']):
            for token in doc:
                if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha):
                    preprocessed_tokens.append(token_representation(token))

    else:
        for doc in nlp.pipe(tokens, disable=['parser', 'ner']):
            for token in doc:
                if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha) and token.pos_ in pos_filter:
                    preprocessed_tokens.append(token_representation(token))
    return preprocessed_tokens


def to_vecs(tokens = List[str], use_phrases: bool = False) -> List[float]:
    if use_phrases:
        bigram_transformer = Phrases(tokens)
        model = Word2Vec(bigram_transformer[tokens], size=100, window=10, min_count=1, workers=4)
    else:
        model = Word2Vec(tokens, size=300, window=10, min_count=1, workers=4, iter=15)
    print(model.wv.most_similar("tumor"))
    return model.wv

tokens = preprocess(tokens_from_file(path="E:/CPG-AMIA2020/Plain Text/cpg-tokens.txt")[:100000000])
print(tokens[:10])
to_vecs([tokens])