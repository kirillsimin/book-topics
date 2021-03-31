import os
import codecs
import string

import pickle
import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords

from gensim import corpora, models

lemmatizer = WordNetLemmatizer()
lda = models.ldamodel.LdaModel

manual = ["could", "would", '"','“', '”', "’", "iii",'———', '--', 'mr.', '–', 'said', 'one', '‘', 'p.', 'pp', '...', '....', 'little', 'much', 'new', 'good', 'old', 'like', 'many', 'big', 'way', 'next', 'last', 'org', 'http', 'mrs.',' le', 'say', 'even', 'back', 'thing', 'got', 'get', 'two', 'may', 'say', 'want', 'know', 'see', 'al.', 'n\'t', '\'ll', '\'ve', '\'re', 'ltd.', 'come', 'chapter', 'also']

exclude = set(stopwords.words('english') + list(string.punctuation) + manual)

directory = './data/'
num_topics = 7
passes = 20
alpha = 'auto'
max_books = 100

books = []

def file_to_text(path):
    f = codecs.open(path, 'r', errors='replace')
    text = f.read()#.lower()
    f.close()
    return text

def text_to_lemmas(text=''):
    sentences = sent_tokenize(text)
    tokens = []
    for sentence in sentences:
        sent_tokens = pos_tag(word_tokenize(sentence))
        tokens = tokens + sent_tokens

    lemmas = []
    
    for token in tokens:
        token_pos = token[1]
        word = lemmatizer.lemmatize(token[0].lower())

        if word not in exclude and len(word) > 2 and token_pos in ['NN', 'JJ']:
            lemmas.append(word)
    
    return lemmas

def build_lda_modal(lemmas):
    dictionary = corpora.Dictionary(lemmas)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in lemmas]
    ldamodel = lda(
        corpus=doc_term_matrix,
        #alpha=alpha,
        num_topics=num_topics,
        passes=passes,
        id2word=dictionary,
        minimum_probability=0.4,
        per_word_topics=False
    )

    return ldamodel, doc_term_matrix
    
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

def topics_to_dict(topics):
    topics = dict([])

    for topic_group in topic_groups:
        topic_group = topic_group[1].split(' + ')
        for topic in topic_group:
            topic = topic.split('*')
            word = topic[1].strip('"')
            weight = float(topic[0])
            
            if weight > 0.00 and weight > topics.get(word, 0):
                topics[word] = weight

    topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
    return topics


def text_files_to_wordbags():
    count = 0
    files = os.listdir(directory)
    files.sort()

    words = []
    titles = []

    for filename in files:
        if count < max_books:
            if '.txt' in filename:
                count += 1
                book_no = filename.split(' -- ')[0]
                book_title = filename.split(' -- ')[1].strip('.txt')
                titles.append(book_title)
                print('\n' + book_title)
                 
                text = file_to_text(directory + '/' + filename)
                lemmas = text_to_lemmas(text)
                words.append(lemmas)
    return words, titles

# generate word bags and titles
#words, titles = text_files_to_wordbags()
#words_df = pd.DataFrame(words).to_pickle('./results/books_words.pkl')
#titles_df = pd.DataFrame(titles).to_pickle('./results/book_titles.pkl')



# read generated word bags and titles
titles = pd.read_pickle('./results/book_titles.pkl').values.tolist()
words = pd.read_pickle('./results/books_words.pkl').values.tolist()
words = [list(filter(None, doc)) for doc in words]


# generate topics
ldamodel, corpus = build_lda_modal(words)
topic_groups = get_topics(ldamodel)
transformed_corpus = ldamodel[corpus]

print(alpha, passes)

for topic in topic_groups:
    print(topic)

for topic in transformed_corpus:
    print(topic)


