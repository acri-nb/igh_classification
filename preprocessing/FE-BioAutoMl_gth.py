import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import argparse
import subprocess
import shutil
import sys
import os.path
import time


def feature_extraction(fasta_to_extract, fasta_label, features, foutput):

	"""Extracts the features from the sequences in the fasta files."""

	path = foutput + '/feat_extraction'
	path_results = foutput

	try:
		shutil.rmtree(path)
		shutil.rmtree(path_results)
	except OSError as e:
		print("Error: %s - %s." % (e.filename, e.strerror))
		print('Creating Directory...')

	if not os.path.exists(path_results):
		os.mkdir(path_results)

	if not os.path.exists(path):
		os.mkdir(path)

	datasets = []
	fasta_list = []

	print('Extracting features with MathFeature...')

	file = fasta_to_extract.split('/')[-1]
	preprocessed_fasta = path + '/pre_' + file
	subprocess.run(['python', 'MathFeature/preprocessing/preprocessing.py',
					'-i', fasta_to_extract, '-o', preprocessed_fasta],
					stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

	fasta_list.append(preprocessed_fasta)

	if 1 in features:
		dataset = path + '/NAC.csv'
		subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py',
						'-i', preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-t', 'NAC', '-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 2 in features:
		dataset = path + '/DNC.csv'
		subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-t', 'DNC', '-seq', '1'], stdout=subprocess.DEVNULL,
						stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 3 in features:
		dataset = path + '/TNC.csv'
		subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-t', 'TNC', '-seq', '1'], stdout=subprocess.DEVNULL,
						stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 4 in features:
		dataset_di = path + '/kGap_di.csv'
		dataset_tri = path + '/kGap_tri.csv'

		subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
						preprocessed_fasta, '-o', dataset_di, '-l',
						fasta_label, '-k', '1', '-bef', '1',
						'-aft', '2', '-seq', '1'],
						stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

		subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
						preprocessed_fasta, '-o', dataset_tri, '-l',
						fasta_label, '-k', '1', '-bef', '1',
						'-aft', '3', '-seq', '1'],
						stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset_di)
		datasets.append(dataset_tri)

	if 5 in features:
		dataset = path + '/ORF.csv'
		subprocess.run(['python', 'MathFeature/methods/CodingClass.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label],
						stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 6 in features:
		dataset = path + '/Fickett.csv'
		subprocess.run(['python', 'MathFeature/methods/FickettScore.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 7 in features:
		dataset = path + '/Shannon.csv'
		subprocess.run(['python', 'MathFeature/methods/EntropyClass.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-k', '5', '-e', 'Shannon'],
						stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 8 in features:
		dataset = path + '/FourierBinary.csv'
		subprocess.run(['python', 'MathFeature/methods/FourierClass.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 9 in features:
		dataset = path + '/FourierComplex.csv'
		subprocess.run(['python', 'other-methods/FourierClass.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-r', '6'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 10 in features:
		dataset = path + '/Tsallis.csv'
		subprocess.run(['python', 'other-methods/TsallisEntropy.py', '-i',
						preprocessed_fasta, '-o', dataset, '-l', fasta_label,
						'-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		datasets.append(dataset)

	if 11 in features:
		dataset = path + '/Chaos.csv'
		# classifical_chaos(preprocessed_fasta, fasta_label, 'Yes', dataset)
		datasets.append(dataset)

	if 12 in features:
		dataset = path + '/BinaryMapping.csv'

		text_input = fasta_to_extract + '\n' + fasta_label + '\n'

		subprocess.run(['python', 'MathFeature/methods/MappingClass.py',
						'-n', '1', '-o',
						dataset, '-r', '1'], text=True, input=text_input,
					   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

		with open(dataset, 'r') as temp_f:
			col_count = [len(l.split(",")) for l in temp_f.readlines()]

		colnames = ['BinaryMapping_' + str(i) for i in range(0, max(col_count))]

		df = pd.read_csv(dataset, names=colnames, header=None)
		df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
		df.to_csv(dataset, index=False)
		
		datasets.append(dataset)

	"""Concatenating all the extracted features"""

	if datasets:
		datasets = list(dict.fromkeys(datasets))
		dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
		dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
		dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

	dataframes.pop('nameseq')
	dataframes.pop('label')
	
	fextracted = path + '/features_extracted.csv'
	dataframes.to_csv(fextracted, index=False)

	return fextracted

##########################################################################
##########################################################################


if __name__ == '__main__':
	print('\n')
	print('###################################################################################')
	print('###################################################################################')
	print('##########         BioAutoML- Feature Extraction                        ###########')
	print('##########              Author: gth Adaptatiton                         ###########')
	print('###################################################################################')
	print('###################################################################################')
	print('\n')
	parser = argparse.ArgumentParser()
	parser.add_argument('-fasta_to_extract', '--fasta_to_extract', required=True,
						help='fasta format file, e.g., fasta/ncRNA.fasta')
	parser.add_argument('-fasta_label', '--fasta_label', required=True,
						help='label for fasta files, e.g., ncRNA')
	parser.add_argument('-output', '--output', required=True, help='results directory, e.g., result/')

	args = parser.parse_args()
	fasta_to_extract = args.fasta_to_extract
	fasta_label = args.fasta_label
	foutput = str(args.output)

	if os.path.exists(fasta_to_extract) is True:
		print('fasta_to_extract - %s: Found File' % fasta_to_extract)
	else:
		print('fasta_to_extract - %s: File not exists' % fasta_to_extract)
		sys.exit()

	start_time = time.time()

	features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	fextracted = feature_extraction(fasta_to_extract, fasta_label, features, foutput)

	cost = (time.time() - start_time) / 60
	print('Computation time - Feature Extraction: %s minutes' % cost)


##########################################################################
##########################################################################