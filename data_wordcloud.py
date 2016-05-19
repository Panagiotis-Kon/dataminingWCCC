#!/usr/bin/env python2
"""
Generating a square wordcloud from the given dataset.
"""

import pandas as pd
from os import path
from wordcloud import WordCloud
import sys

def wordcloud_generator(argv):
	print("Loading file: %s" % str(argv))
	dataset=pd.read_csv(argv,sep='\t')
	print("Loading finished.")
	img_w=960
	img_h=540
	relative_sc=1
	wordcloud = WordCloud(width=img_w, height=img_h, relative_scaling=relative_sc)
	print
	print("Creating categories' list.")
	list_categories ={}
	for index, row in dataset.iterrows():
		try:
			list_categories[row['Category']] += row['Content']
		except KeyError:
		   	list_categories[row['Category']] = row['Content']
	print("Categories created.")
	print
	print("Creating word clouds.")
	for category, words in list_categories.iteritems():
		wordcloud.generate(words)
		image = wordcloud.to_image()
		image.save("data/Wordcloud_" + category + "_" + str(img_w) + "x" + str(img_h) + ".png")
		image.show()
	print("Word Clouds finished.")
	return

print"Program starts..."
wordcloud_generator(sys.argv[1])
print"Program ends..."
