from gensim.models.keyedvectors import KeyedVectors

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
from pylab import figure
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

print('Loading model...')
glove_model = KeyedVectors.load_word2vec_format("./models/gensim_glove_vectors.txt", binary=False)

print('Opening wordlist...')
words = open('./results/words.txt', 'r').read().split()

def words_coordinates(word_list=words, word_vectors = glove_model):
    word_plus_coordinates=[]

    print('Adding vectors...')
    for word in word_list:
        try:
            current_row = []
            current_row.append(word)
            current_row.extend(word_vectors[word])
            word_plus_coordinates.append(current_row)
        except:
            print('    ' + word + ' was not found in model.')


    return pd.DataFrame(word_plus_coordinates)


def plot_3d():
    words_coords = words_coordinates()
    # 3D chart
    coords = TSNE(n_components=3).fit_transform(words_coords.iloc[:, 1:50])
    coords = pd.DataFrame(coords, columns=['x','y','z'])
    coords['word'] = words_coords.iloc[:, 0]

    
    fig = plt.figure(figsize=(35,35))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coords['x'], coords['y'], coords['z'], c='r', marker='o')

    for i in range(len(coords)):
        ax.text(coords['x'][i], coords['y'][i], coords['z'][i], coords['word'][i])
    
    plt.savefig('./results/book-topics-scatter-3d.png')


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
    
    #p1 = sns.scatterplot(data=coords, x=x, y=y)
    [plt.scatter(coords[x][i], coords[y][i], s=coords['count'][i]*10, c='c') for i in range(len(coords))]

    [plt.text(coords[x][i], coords[y][i], coords[label][i]) for i in range(len(coords))]
    plt.savefig('./results/book-topics-scatter-2d.png')


plot_2d()


