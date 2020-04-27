from typing import List
import spacy
from gensim import models as gensim
from gensim.models import Word2Vec, FastText, Phrases
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath

def tokens_from_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = f.read()

    return data.split("\n")


def preprocess(tokens: List[str], lemmatize: bool = True, lower: bool = False,
               pos_filter: list = None, remove_stopwords: bool = True,
               remove_punctuation: bool = False, lan_model=None) -> List[str]:
    def token_representation(token):
        representation = str(token.lemma_) if lemmatize else str(token)
        if lower:
            representation = representation.lower()
        return representation

    nlp = spacy.load("de_core_news_sm") if lan_model is None else lan_model
    nlp.Defaults.stop_words |= {"der", "die", "das", "Der", "Die", "Das", "bei", "Bei", "In", "in"}

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
                if (not remove_stopwords or not token.is_stop) and (
                        not remove_punctuation or token.is_alpha) and token.pos_ in pos_filter:
                    preprocessed_tokens.append(token_representation(token))
    return preprocessed_tokens


def to_vecs(tokens=List[str], use_phrases: bool = False, w2v_model=gensim.Word2Vec) -> gensim.KeyedVectors:
    if use_phrases:
        bigram_transformer = Phrases(tokens)
        model = w2v_model(bigram_transformer[tokens], size=100, window=10, min_count=1, workers=4)
    else:
        model = Word2Vec(tokens, size=300, window=5, min_count=1, workers=4, iter=10)
    return model.wv


def save_vecs(word_vectors: gensim.KeyedVectors, path="E:/AML4DHGermanVecs/test_vecs.kv"):
    word_vectors.save(get_tmpfile(path))


def load_vecs(path="E:/AML4DHGermanVecs/test_vecs.kv") -> gensim.KeyedVectors:
    return gensim.KeyedVectors.load(path)


tokens = preprocess(tokens_from_file(path="E:/CPG-AMIA2020/Plain Text/cpg-tokens.txt")[:1000])
print(tokens[:10])
vecs = to_vecs([tokens], use_phrases=True)
print(vecs.most_similar("Tumor"))
save_vecs(vecs)

# vecs = load_vecs()
# print(vecs.most_similar("Tumor"))