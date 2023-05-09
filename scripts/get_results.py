import os
import glob
import shutil
from argparse import ArgumentParser


def arg_parser():
	"""
    Defining parser for options in command line
    Returns the parser
    """
	parser = ArgumentParser(description='Get PDF results from folder --input to '
										'folder --output')

	parser.add_argument("-i, --i",
						dest="input",
						action="store",
						type=str,
						help="Input directory where to extract results",
						required=True)
	parser.add_argument('-o, --output',
						dest='output',
						action="store",
						type=str,
						required=True,
						help="output directory to save results")

	return parser.parse_args()


input_dir = f'{arg_parser().input}*'
output_dir = arg_parser().output
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
results = list()
folders_without_result = list()
# Get results for all foo/output.csv files inside input_dir
# and copy PDF reports to output dir
for folder in sorted(glob.glob(input_dir)):
	f_name = folder.split("/")[-1]
	if os.path.isdir(folder) and len(os.listdir(folder)) != 0:
		# Check if the output.csv file exists,
		# meaning that the estimated distance has been calculated
		if os.path.exists(f'{folder}/output/results/output.csv'):
			result = open(f'{folder}/output/results/output.csv', 'r').readlines()[1].strip() + f',{f_name}\n'
			results.append(result)
			shutil.copyfile(f'{folder}/output/figures/{f_name}.pdf', f'{output_dir}/{f_name}.pdf')
		else:
			folders_without_result.append(f'{f_name}')
print(f'\nCreating {output_dir}results.csv file...\n')

# Create results.csv in output dir
with open(output_dir + 'results.csv', 'w') as rf:
	rf.write('mu (nm),SErr (nm),Sigma (nm),SErr (nm),Number of measurements,NAME\n')
	for r in results:
		rf.write(r)
if len(folders_without_result) != 0:
	print(f'Folders without result: {",".join(folders_without_result)}\n####DONE!\n')
else:
	print('Folders without result: 0\n####DONE!\n')
