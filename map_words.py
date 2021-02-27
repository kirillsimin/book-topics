import matplotlib.pyplot as plt
from pylab import figure
import pandas as pd

data = pd.read_pickle('./results/coords_and_embeddings.pkl')

def plot_3d(data):

    fig = plt.figure(figsize=(50,50))

    ax = fig.add_subplot(projection='3d')

    graph = ax.scatter(data['book_num'], data['x'], data['y'], s=100, c=data['book_num'], marker='o')

    print('Adding labels...')
    for i in range(len(data)):
        ax.text(data['book_num'][i], data['x'][i], data['y'][i], data['word'][i], alpha=0.75 )

    #ax.view_init(90,0)

    plt.savefig('./results/book-topics-scatter-3d.png')
    print('3D chart image saved.')
    

def plot_2d(data):
    word_count = data.value_counts('word').rename('count').to_frame().reset_index()
    data = data.groupby('word', as_index=False).first()
    data = data.merge(word_count)

    
    print(data) 
    plt.figure(figsize = (55,55))
    
    [plt.scatter(data['x'][i], data['y'][i], s=pow(int(data['count'][i]),2), c='c') for i in range(len(data))]

    [plt.text(data['x'][i], data['y'][i], data['word'][i]) for i in range(len(data))]
    plt.savefig('./results/book-topics-scatter-2d.png')
    
    

plot_2d(data)
