import pandas as pd
from six.moves import cPickle as pickle

colnames = ['heading', 'link', 'topic', 'source', 'content']
df1=pd.read_csv("/home/sarthak/Mydata/Projects/silicon-beach-data/urls/test2.csv", names=colnames)

contentList = df1.content.tolist()
headList = df1.heading.tolist()

pickleFile = '/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/article_and_heading_data.pickle'
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


