import os
import codecs
import string

import pickle
import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords

from gensim import corpora, models

lemmatizer = WordNetLemmatizer()
lda = models.ldamodel.LdaModel

manual = ["could", "would", '"','“', '”', "’", "iii",'———', '--', 'mr.', '–', 'said', 'one', '‘', 'p.', 'pp', '...', '....', 'little', 'much', 'new', 'good', 'old', 'like', 'many', 'big', 'way', 'next', 'last', 'org', 'http', 'mrs.',' le', 'say', 'even', 'back', 'thing', 'got', 'get', 'two', 'may', 'say', 'want', 'know', 'see', 'al.', 'n\'t', '\'ll', '\'ve', 'come', 'chapter', 'also']

exclude = set(stopwords.words('english') + list(string.punctuation) + manual)

directory = './data'
num_topics = 10
passes = 7
count = 0
max_books = 100

books = []

def file_to_text(path):
    f = codecs.open(path, 'r', errors='replace')
    text = f.read().lower()
    f.close()
    return text

def text_to_lemmas(text='', pos=None):
    tokens = word_tokenize(text)
    lemmas = []
    
    for token in tokens:
        tagged = pos_tag([token])
        token_pos = tagged[0][1]
        word = lemmatizer.lemmatize(tagged[0][0])

        if word not in exclude and len(word) > 2 and token_pos not in ['CC', 'CD']:
            if pos is None:
                lemmas.append(word)
            elif token_pos == pos:
                lemmas.append(word)

    return lemmas

def build_lda_modal(lemmas, num_topics = num_topics):
    dictionary = corpora.Dictionary(lemmas)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in lemmas]
    ldamodel = lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes, minimum_probability=0.9)

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

def topics_to_dict(topics):
    topics = dict([])

    for topic_group in topic_groups:
        topic_group = topic_group[1].split(' + ')
        for topic in topic_group:
            topic = topic.split('*')
            word = topic[1].strip('"')
            weight = float(topic[0])
            
            if weight > 0.003 and weight > topics.get(word, 0):
                topics[word] = weight

    topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
    return topics

files = os.listdir(directory)
files.sort()

book_topics = []

for filename in files:
    if count < max_books:
        if '.txt' in filename:
            count += 1
            print('\n')
            print('\n' + filename)
            print('\n')
             
            text = file_to_text(directory + '/' + filename)
            lemmas = text_to_lemmas(text)
            lemmas = np.array_split(lemmas, 15)

            ldamodel = build_lda_modal(lemmas)
            topic_groups = get_topics(ldamodel)

            topics = topics_to_dict(topic_groups)
            print(topics)
            topic_words = [topic[0] for topic in topics]

            book_no = filename.split(' -- ')[0]
            book_title = filename.split(' -- ')[1].strip('.txt')

            book_topics.append([book_no, book_title, topics, topic_words])


print()

df = pd.DataFrame(np.array(book_topics, dtype="object"), columns=['number', 'title', 'topics', 'topic_words'])
df.to_pickle('./results/topics.pkl')

print(df)
