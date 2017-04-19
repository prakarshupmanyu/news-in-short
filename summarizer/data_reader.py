import pandas as pd
from six.moves import cPickle as pickle

colnames = ['heading', 'link', 'topic', 'source', 'content']
fileName = "/home/prakarsh/Desktop/ouest-france-spider-data_18889-56544_politique.csv"
df1=pd.read_csv(fileName, names=colnames)

contentList = df1.content.tolist()
headList = df1.heading.tolist()

#pickleFile = '/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/article_and_heading_data.pickle'
#pickleFile = '/home/melvin/Documents/USC/news-in-short/DataExtractor/art/lesechos-fr-spider-data_6445-8935_politique.pickle'
pickleFile = '/home/prakarsh/Desktop/ouest-france-spider-data_18889-56544_politique.pickle'

try:
    f = open(pickleFile, 'wb')
    save = {
        'content' : contentList,
        'heading' : headList,

    }
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    f.close()

except Exception as e:
    print('Unable to save pickle file : ' ,e )
