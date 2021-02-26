from gensim.models.keyedvectors import KeyedVectors

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pylab import figure

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

    words = []

    for topic_group in df['topics']:
        book_topics = []
        for topic in topic_group:
            words.append(topic[0])

    return words


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

def prep_data():
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
    data.to_pickle('./results/chart_data.pkl')
    return data

def plot_3d():
    data = pd.read_pickle('./results/chart_data.pkl')
    
    fig = plt.figure(figsize=(50,50))

    ax = fig.add_subplot(projection='3d')

    ax.scatter(data['book_num'], data['x'], data['y'], s=300, c=data['book_num'], marker='o')

    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 0.8, 0.8, 1]))

    print('Adding labels...')
    for i in range(len(data)):
        ax.text(data['book_num'][i], data['x'][i], data['y'][i], data['word'][i], alpha=0.75 )
   
    plt.savefig('./results/book-topics-scatter-3d.png')
    print('3D chart image saved.')
    

def plot_2d():
    x = 'x'
    y = 'y'
    label = 'word'

    word_plus_coordinates = words_coordinates()
    
    coords = TSNE(n_components=2).fit_transform(word_plus_coordinates.iloc[:, 1:50])
    coords = pd.DataFrame(coords, columns=[x, y])

    coords[label] = word_plus_coordinates.iloc[:,0]
    coords['count'] = coords[label].map(coords[label].value_counts())
    coords = coords.drop_duplicates(subset=[label])
    coords = coords.reset_index(drop=True)

    print(coords)
    
    plt.figure(figsize = (55,55))
    
    [plt.scatter(coords[x][i], coords[y][i], s=pow(coords['count'][i],2.5), c='c') for i in range(len(coords))]

    [plt.text(coords[x][i], coords[y][i], coords[label][i]) for i in range(len(coords))]
    plt.savefig('./results/book-topics-scatter-2d.png')


plot_3d()


