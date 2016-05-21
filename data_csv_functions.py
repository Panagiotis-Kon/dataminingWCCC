def import_from_csv(file_name):
	print("Loading file: %s" % str(file_name))
	dataset=pd.read_csv(file_name, sep='\t')
	print("Loading finished.")
	return dataset

def export_to_csv(file_name,data):
	print("Exporting to file: %s" % str(file_name))
    dataset = pd.DataFrame.from_dict(data, orient='index')
    dataset.to_csv(file_name, sep='\t', na_rep='0.0')
	print("Exporting finished.")
	return