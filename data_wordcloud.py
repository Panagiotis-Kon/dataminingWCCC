#!/usr/bin/env python2
# Based on algorithms from:
# https://github.com/amueller/word_cloud
"""
Generating a square wordcloud from the given dataset.
"""

import pandas as pd
from os import path
from wordcloud import WordCloud
import sys
import data_csv_functions as dcsv

def wordcloud_generator(dataset):
	img_w=960
	img_h=540
	relative_sc=1
	wordcloud = WordCloud(width=img_w, height=img_h, relative_scaling=relative_sc)
	print("Creating categories' list.")
	sys.stdout.write("Processing: ")
	list_categories ={}
	count=0.0
	for index, row in dataset.iterrows():
		if count==500:
			count=0
			# update the bar
			sys.stdout.write("#")
			sys.stdout.flush()
		else:
			count=count+1
		try:
			list_categories[row['Category']] += row['Content']
		except KeyError:
		   	list_categories[row['Category']] = row['Content']
	print
	print("Categories created.")
	print('+' * 60)
	print("Creating word clouds.")
	print('-' * 60)
	for category, words in list_categories.iteritems():
		# update for process
		print("Creating Category: "+category)
		wordcloud.generate(words)
		image = wordcloud.to_image()
		image.save("./data/Wordcloud_" + category + "_" + str(img_w) + "x" + str(img_h) + ".png")
		image.show()
	print('-' * 60)
	print("WordClouds' creation finished.")
	return

# The main of the program start here #
if __name__ == "__main__":
	print"   WordCloud Program starts..."
	print('=' * 60)
	dataset=dcsv.import_from_csv(sys.argv[1])
	wordcloud_generator(dataset)
	print('=' * 60)
	print"Program ends..."