#!/usr/bin/env python2
"""
Generating a square wordcloud from the given dataset.
"""

import pandas as pd
from os import path
#from wordcloud import WordCloud
import sys

def main(argv):
	print("Loading file: %s" % str(argv))
	dataset=pd.read_csv(argv,sep='\t')
	print(dataset.columns)
	print("Loading finished")
	print(dataset["Category"])
	wordcloud = WordCloud(max_font_size=40, relative_scaling=.5)
	list_categories ={}
	for index, row in dataset.iterrows():
		try:
			list_categories[row['Category']] += row['Content']
		except KeyError:
		   	list_categories[row['Category']] = row['Content']
	for category, words in list_categories.iteritems():
		wordcloud.generate(words)
		image = wordcloud.to_image()
		image.save("Wordcloud_" + category + ".png")
		image.show()
	return

print"Program starts..."
main(sys.argv[1])
print"Finished"
