"""
Copyright 2012 Xixuan (Aaron) Feng and Arun Kumar and Christopher Re

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

##########################################################################
## input CRF++ file format: http://crfpp.sourceforge.net/#usage
##########################################################################

import sys
import re

def output_dict(dict, outputFile) :
	fout = open(outputFile, 'w')
	for k, v in dict.items() :
		print >> fout, '{0}\t{1}'.format(k, v)
	fout.close()

def collect_label_set(dataFile) :
	fin = open(dataFile, 'r')
	label2id = {}
	labelCnt = 0
	for line in fin:
		line = line.rstrip('\n')
		if line != '':
			label = line.rsplit(' ',1)[-1]
			if label not in label2id :
				label2id[label] = labelCnt
				labelCnt = labelCnt + 1
	fin.close()
	return label2id

xRegex = re.compile('%x\[(-?\d+),(\d+)]')

def read_template(templateFile) :
	uLines = []
	bLines = []
	fin = open(templateFile, 'r')
	for line in fin :
		line = line.rstrip('\n')
		if line.startswith('#') or line == '' :
			pass
		elif line.startswith('U') :
			uLines.append(line)
		elif line.startswith('B') :
			bLines.append(line)
		else :
			print >> sys.stderr, 'wrong template file format: ^%s' % line
			sys.exit(2)
	fin.close()
	return (uLines, bLines)

def get_repl(featuresBag, row, col) :
	nTokens = len(featuresBag)
	if row < 0 :
		return 'B[%d,%d]' % (row, col)
	elif row >= nTokens :
		return 'E[%d,%d]' % (row - nTokens, col)
	else :
		return featuresBag[row][col]

def expand_one(line, featuresBag, i) :
	obs = str(line)
	m = xRegex.search(obs)
	while m is not None :
		deltaRow = int(m.group(1))
		row = deltaRow + i
		col = int(m.group(2))
		repl = get_repl(featuresBag, row, col)
		obs = xRegex.sub(repl, obs, 1) 
		m = xRegex.search(obs)
	return obs

def fill_o2i_doc(featuresBag, uLines, bLines, obs2id, nLabels, F) :
	nTokens = len(featuresBag)
	for i in range(nTokens) :
		for uline in uLines :
			obs = expand_one(uline, featuresBag, i)
			if obs not in obs2id:
				obs2id[obs] = F
				F = F + nLabels
		for bline in bLines :
			obs = expand_one(bline, featuresBag, i)
			if obs not in obs2id:
				obs2id[obs] = F
				F = F + nLabels * nLabels
	return obs2id, F

def fill_o2i_all(dataFile, uLines, bLines, nLabels) :
	obs2id = {}
	fin = open(dataFile, 'r')
	F = 0
	featuresBag = []
	for line in fin :
		line = line.rstrip('\n')
		if line == '' :
			obs2id, F = fill_o2i_doc(featuresBag, 
					uLines, bLines, obs2id, nLabels, F)
			featuresBag = []
		else :
			features = line.split(' ')
			features.pop()
			featuresBag.append(features)
	fin.close()
	return obs2id, F

def expand_doc(featuresBag, uLines, bLines, obs2id) :
	uobsBag = []
	bobsBag = []
	nTokens = len(featuresBag)
	for i in range(nTokens) :
		for uline in uLines :
			obs = expand_one(uline, featuresBag, i)
			if obs not in obs2id:
				uobsBag.append('-1')
			else :
				uobsBag.append(str(obs2id[obs]))
		for bline in bLines :
			obs = expand_one(bline, featuresBag, i)
			if obs not in obs2id:
				bobsBag.append('-1')
			else :
				bobsBag.append(str(obs2id[obs]))
	return (uobsBag, bobsBag)

def output_as_obs_labels(dataFile, label2id, obs2id, uLines, bLines, outputFile):
	fin = open(dataFile, 'r')
	fout = open(outputFile, 'w')
	exCnt = 0
	featuresBag = []
	labelsBag = []
	for line in fin:
		line = line.rstrip('\n')
		if line == '':
			exCnt = exCnt + 1
			(uobsBag, bobsBag) = expand_doc(featuresBag, 
					uLines, bLines, obs2id)
			print >> fout, '{%s}\t{%s}\t{%s}' % (
					','.join(uobsBag),
					','.join(bobsBag),
					','.join(labelsBag))
			featuresBag = []
			labelsBag = []
		else :
			features = line.split(' ')
			label = features.pop()
			labelId = label2id[label]
			featuresBag.append(features)
			labelsBag.append(str(labelId))
	fin.close()
	fout.close()
	return exCnt

def main() :
	# get file names from command-line
	dataFile = sys.argv[1]
	templateFile = sys.argv[2]
	labelFile = sys.argv[3]
	obsFile = sys.argv[4]
	bismarckFile = sys.argv[5]

	# hash table 1: map each text label to a distinct label id
	label2id = collect_label_set(dataFile)
	nLabels = len(label2id)
	print 'Number of labels: %d' % nLabels
	output_dict(label2id, labelFile)

	# hash table 2: map each observation to a distict id
	#   An observation is an instantiation of a line in template
	#   in a position in a document. One observation instantiated by
	#   unigram lines corresponds to # labels indices in weight vector,
	#   and bigram (# labels)^2.
	(uLines, bLines) = read_template(templateFile)
	obs2id, F = fill_o2i_all(dataFile, uLines, bLines, nLabels)
	print 'Number of features: %d' % F
	output_dict(obs2id, obsFile)

	# last pass of data file to generate file for copying yo table
	output_as_obs_labels(dataFile, label2id, obs2id, uLines, bLines,
			bismarckFile)

if __name__ == '__main__':
	if (len(sys.argv) != 6) :
		print 'Usage: python crf_data_prepare.py [crf++ format file] \
[template file] [label output file] [observation output file] \
[data output for Bismarck]'
	else :
		main()

