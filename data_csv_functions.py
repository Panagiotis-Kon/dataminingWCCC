#!/usr/bin/env python2
import pandas as pd

def import_from_csv(file_name):
	print("Loading file: %s" % str(file_name))
	dataset=pd.read_csv(file_name, sep='\t')
	print("Loading finished.")
	return dataset

def export_to_csv(file_name,data):
	print("Exporting to file: %s" % str(file_name))
	dataset = pd.DataFrame.from_dict(data, orient='index')
	dataset.to_csv(file_name, sep='\t', na_rep='0.00', encoding='utf-8', dialect='excel')
	print("Exporting finished.")
	return

def export_to_csv2(file_name,data):
	print("Exporting to file: %s" % str(file_name))
	df=pd.DataFrame(data,index=['Politics','Football','Technology','Film','Business'])
	df = pd.DataFrame(df)
	df.to_csv(file_name, sep='\t', encoding='utf-8')
	print("Exporting finished.")
	return