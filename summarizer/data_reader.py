import pandas as pd
from six.moves import cPickle as pickle

colnames = ['heading', 'link', 'topic', 'source', 'content']
fileName = '/home/prakarsh_upmanyu23/filtered_art/20minutes-fr-Spider-data_26136-34848_politique.csv'
#fileName = "/home/prakarsh_upmanyu23/filtered_art/concatenated_27269.csv"
df1=pd.read_csv(fileName, names=colnames)

contentList = df1.content.tolist()
headList = df1.heading.tolist()

#pickleFile = '/home/sarthak/PycharmProjects/silicon-beachNLP/news-in-short/processedData/article_and_heading_data.pickle'
#pickleFile = '/home/melvin/Documents/USC/news-in-short/DataExtractor/art/lesechos-fr-spider-data_6445-8935_politique.pickle'
pickleFile = '/home/prakarsh_upmanyu23/latribune-fr-Spider-data_0-5848_politique.pickle'
pickleFile = '/home/prakarsh_upmanyu23/output_files/concatenated.pickle'

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
