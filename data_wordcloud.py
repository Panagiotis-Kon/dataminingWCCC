#!/usr/bin/env python2
"""
Generating a square wordcloud from the given dataset.
"""

import pandas as pd
from os import path
from wordcloud import WordCloud
import sys
from data_csv_functions import import_from_csv

def wordcloud_generator(dataset):
	img_w=960
	img_h=540
	relative_sc=1
	wordcloud = WordCloud(width=img_w, height=img_h, relative_scaling=relative_sc)
	print("Creating categories' list.")
	list_categories ={}
	for index, row in dataset.iterrows():
		try:
			list_categories[row['Category']] += row['Content']
		except KeyError:
		   	list_categories[row['Category']] = row['Content']
	print("Categories created.")
	print("Creating word clouds.")
	for category, words in list_categories.iteritems():
		wordcloud.generate(words)
		image = wordcloud.to_image()
		image.save("data/Wordcloud_" + category + "_" + str(img_w) + "x" + str(img_h) + ".png")
		image.show()
	print("WordClouds' creation finished.")
	return

print"Program starts..."
dataset=import_from_csv(sys.argv[1])
wordcloud_generator(dataset)
print"Program ends..."