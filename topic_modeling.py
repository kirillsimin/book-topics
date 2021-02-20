import os
import codecs
import string

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords

from gensim import corpora, models

lemmatizer = WordNetLemmatizer()
lda = models.ldamodel.LdaModel

manual = ["could", "would", '"','“', '”', "’", "iii", '--', 'mr.', '–', 'said', 'one', '‘', 'p.', 'pp', '...', '....', 'little', 'much', 'new', 'good', 'like', 'many', 'big', 'way', 'next', 'last', 'org', 'http', 'mrs.',' le', 'say', 'even', 'back', 'thing', 'got']

exclude = set(stopwords.words('english') + list(string.punctuation) + manual)

directory = './data'
num_topics = 3
count = 0
max_books = 100

def file_to_text(path):
    f = codecs.open(path, 'r', errors='replace')
    text = f.read().lower()
    f.close()
    return text

def text_to_lemmas(text='', pos=None):
    tokens = word_tokenize(text)
    lemmas = []
    for token in tokens:
        if token not in exclude and "'" not in token and "`" not in token and len(token) > 2:
            tagged = pos_tag([token])
            token_pos = tagged[0][1]
            word = tagged[0][0]
            if (pos is not None):
                if (token_pos == pos):
                    lemmas.append(lemmatizer.lemmatize(token))
            else:
                lemmas.append(lemmatizer.lemmatize(token))
    return lemmas

def build_lda_modal(lemmas, num_topics = num_topics):
    dictionary = corpora.Dictionary([lemmas])
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in [lemmas]]
    ldamodel = lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=5, minimum_probability=0)

    return ldamodel
    
def get_topics(ldamodel, num_topics = num_topics):
    topics = ldamodel.print_topics(num_topics=num_topics)
    return topics

def get_topic_words(ldamodel, num_topics = num_topics):
    words = []
    for i in range(0, num_topics):
        topics = ldamodel.show_topic(i)
        for topic in topics:
            words.append(topic[0])

    return words

for filename in os.listdir(directory):
    if count < max_books:
        if '.txt' in filename:
            count += 1
            #print('\n' + filename)
            print('\n')
             
            text = file_to_text(directory + '/' + filename)
            lemmas = text_to_lemmas(text, 'JJ')
            ldamodel = build_lda_modal(lemmas)
            #topics = get_topics(ldamodel)
            topics = get_topic_words(ldamodel)

            for topic in set(topics):
                print(topic, end=' ')
