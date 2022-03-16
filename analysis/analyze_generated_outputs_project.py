"""Script to analyze model's generated outputs."""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
import os

from constants import *
from util import format_score_sentence_output
from collections import Counter
from collections import OrderedDict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def calc_sample_scores(files, first_period=True, score_type='vader'):
	"""Calculate/format scores for samples."""
	scores = []
	lines = []

	for fi_idx, fi in enumerate(files):
		with open(fi, 'r') as f:
			for line in f:
				line = line.strip()
				sample = line.split('\t')[-1]
				if first_period:
					# Cut off the line when we see the first period.
					if '.' in sample:
						period_idx = sample.index('.')
					else:
						period_idx = len(sample)
					sample_end = min(period_idx + 1, len(sample))
				else:
					sample_end = len(sample)
				sample = sample[:sample_end]
				lines.append(sample)

	if score_type == 'textblob':
		for line_idx, line in enumerate(lines):
			blob = TextBlob(line)
			o_score = blob.sentences[0].sentiment.polarity
			scores.append(o_score)
	elif score_type == 'vader':
		def sentiment_analyzer_scores(sent):
			vader_score = analyzer.polarity_scores(sent)
			return vader_score
		analyzer = SentimentIntensityAnalyzer()
		for line_idx, line in enumerate(lines):
			score = sentiment_analyzer_scores(line)
			c = score['compound']
			if c >= 0.05:
				scores.append(1)
			elif c <= -0.05:
				scores.append(-1)
			else:
				scores.append(0)
	elif score_type == 'bert':
		for fi in files:  # Analyze the classifier-labeled samples.
			with open(fi) as f:
				for line in f:
					line = line.strip()
					line_split = line.split('\t')
					score = int(line_split[0])
					scores.append(score)
	else:
		raise NotImplementedError('score_type = textblob, vader, bert')

	assert(len(scores) == len(lines))

	return list(zip(lines, scores))


def plot_scores(scores, ratio=False):
	"""Plot sentiment"""
	width = 0.15
	ind = np.arange(3)

	score_counts = Counter()
	for s in scores:
		if s >= 0.05:
			score_counts['+'] += 1
		elif s <= -0.05:
			score_counts['-'] += 1
		else:
			score_counts['0'] += 1
	if ratio:
		if len(scores):
			score_len = float(len(scores))
			score_counts['+'] /= score_len
			score_counts['-'] /= score_len
			score_counts['0'] /= score_len
	ordered_score_counts = [round(score_counts['-'], 3), round(score_counts['0'], 3),
								round(score_counts['+'], 3)]
	print('# samples: %s, [neg, neu, pos] ratio: %s' % (len(scores), ordered_score_counts))

	# plt.bar(ind + (score_idx * width), ordered_score_counts, width=width, align='edge',
	# 		label=label)
	# plt.xticks(ind + width * 3, ['negative', 'neutral', 'positive'])
	# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, framealpha=0.9)
	# plt.show()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--full_tsv_file',
	                    required=False,
						default='../data/all_generated_output.tsv',
	                    help='TSV file to evaluate.')
	parser.add_argument('--first_period',
	                    required=False,
	                    default=1,
	                    help='Whether to cut samples off after first period.')
	parser.add_argument('--model_type',
	                    required=False,
	                    default='regard2',
						help='`regard2`, `sentiment2`, `regard1` or `sentiment1`.')
	params = parser.parse_args()

	params.first_period = int(params.first_period) == 1

	print('params', params)

	# Format BERT outputs.
	dir_name = os.path.dirname(params.full_tsv_file)
	base_name = os.path.basename(params.full_tsv_file)
	pred_file = os.path.join(dir_name, params.model_type + '_' + base_name + '_preds.tsv')
	new_lines = format_score_sentence_output(params.full_tsv_file, pred_file)
	labeled_file = os.path.join(dir_name, params.model_type + '_' + base_name + '_labeled.tsv')
	with open(labeled_file, 'w') as o:
		o.write('\n'.join(new_lines))

	sample_to_score = calc_sample_scores([labeled_file],
	                                     first_period=params.first_period,
	                                     score_type='bert')

	# sample_to_score is a 2D list of [[line, score]]

	scores = sample_to_score[:][1]
	plot_scores(scores, ratio=True)


if __name__ == '__main__':
	main()
