from gensim.models.keyedvectors import KeyedVectors

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

from tqdm import tqdm


def get_model():
    print('Loading model...')
    return KeyedVectors.load_word2vec_format("./models/gensim_glove_vectors.txt", binary=False)

def get_words():
    print('Opening wordlist...')
    df = pd.read_pickle('./results/topics.pkl')
    return df

    '''
    words = []

    for topic_group in df['topics']:
        book_topics = []
        for topic in topic_group:
            words.append(topic[0])

    return words
    '''


def words_coordinates(word_list, word_vectors):
    word_plus_coordinates=[]

    for word in word_list:
        try:
            current_row = []
            current_row.append(word)
            current_row.extend(word_vectors[word])
            word_plus_coordinates.append(current_row)
        except:
            print('    ' + word + ' was not found in model.')


    return pd.DataFrame(word_plus_coordinates)

def add_embeddings():
    glove_model = get_model()
    df = get_words()

    data = pd.DataFrame(columns=['x', 'y', 'book_num', 'word', 'coords'])

    print('Transposing topic words into rows...')

    for i, row in tqdm(df.iterrows()):
        for word in row['topic_words']:
            data = data.append({
                'book_num' : int(row['number']),
                'word': word,
            }, ignore_index=True)

    
    print('Adding word embeddings...')

    for i, row in data.iterrows():
        try:
            coords =  glove_model[row['word']]
            data.at[i, 'coords'] = coords
        except:
            data = data.drop(i)

            #print('    ' + row['word'] + ' not in model.')
    data = data.reset_index() 
    del data['index']
    
    print('Reducing dimensions...')
    
    flat_coords = TSNE(n_components=2).fit_transform(data.coords.tolist())

    for i,row in data.iterrows():
        data.at[i, 'x'] = flat_coords[i][0]
        data.at[i, 'y'] = flat_coords[i][1]

    print(data)
    data.to_pickle('./results/coords_and_embeddings.pkl')
    return data

add_embeddings()


