import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from PIL import Image
import numpy as np


def generate_wordcloud(dataset):
    for word in stopwords.words("english"):
        STOPWORDS.add(word)
    mask = np.array(Image.open("mask.jpg"))
    wc = WordCloud(background_color='white', width=800, height=600, stopwords=STOPWORDS, relative_scaling=1, mask=mask)
    categories ={}
    for index, row in dataset.iterrows():
        try:
            categories[row['Category']] += row['Content']
        except KeyError:
            categories[row['Category']] = row['Content']
    for category, words in categories.iteritems():
        wc.generate(words)
        image = wc.to_image()
        image.save("Wordcloud_" + category + ".png")
    return


print "Processing"
generate_wordcloud(pd.read_csv('DataSets/train_set.csv', sep='\t'))
print "Done"
