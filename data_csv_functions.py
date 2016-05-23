#!/usr/bin/env python2
# Functions to import and export csv files
import pandas as pd

def import_from_csv(file_name):
	print('=' * 60)
	print("Loading file: %s" % str(file_name))
	dataset=pd.read_csv(file_name, sep='\t')
	print("Loading finished.")
	print('=' * 60)
	return dataset

def export_to_csv_categories(file_name,data):
	print('=' * 60)
	print("Exporting to file: %s" % str(file_name))
	dataset = pd.DataFrame(data)
	dataset.to_csv(file_name, sep='\t')
	print("Exporting finished.")
	print('=' * 60)
	return

def export_to_csv_cluster(file_name,data):
	print('=' * 60)
	print("Exporting to file: %s" % str(file_name))
	dataset = pd.DataFrame.from_dict(data, orient='index')
	dataset.to_csv(file_name, sep='\t', na_rep='0.00', float_format='%.2f', encoding='utf-8', dialect='excel')
	print("Exporting finished.")
	print('=' * 60)
	return

def export_to_csv_statistic(file_name,data):
	print('=' * 60)
	print("Exporting to file: %s" % str(file_name))
	dataset = pd.DataFrame.from_dict(data, orient='index')
	dataset.to_csv(file_name, sep='\t', encoding='utf-8', float_format='%.3f', index_label="Statistic Measure")
	print("Exporting finished.")
	print('=' * 60)
	return