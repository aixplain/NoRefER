import torch, re, pdb
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModel
import evaluate
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
parser = ArgumentParser(description='Input parameters')
parser.add_argument('--filename', default='en-libre.csv', type=str, help='Enter name of the .csv file filename e.g. en-libre.csv')
args = parser.parse_args()


perplexity = evaluate.load("perplexity", module_type="metric")

print('Producing the Results for %s\nUse --filename to change the file.' % args.filename)

file_path = "./dataset/" + args.filename
data = pd.read_csv(file_path)


data = data[data['WER, uncased, not punctuated'].notna()]

data['WER, uncased, not punctuated'] = data['WER, uncased, not punctuated'].str.strip('%').astype(float)

sentences = data['outputText'].astype(str).to_list()

sentences = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]

results = perplexity.compute(model_id='nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large',
                                        add_start_token=True,
                                        predictions=sentences)
data['preds'] = results['perplexities']
#data['preds'] = -data['preds']

def meanRanking(met):
    groups = data.groupby('inputText')['preds']
    data['pred_rank'] = groups.rank(method=met, ascending=True)
    print(data.groupby('Provider')['pred_rank'].mean())

meanRanking("average")

data['wer_rank'] = data.groupby('inputText')['WER, uncased, not punctuated'].rank(method="average", ascending=False)

from scipy import stats

print('Correlations with the WER rankings')
print('pearson %f' % stats.pearsonr(data['wer_rank'], data['pred_rank'])[0])
print('spearman %f' % + stats.kendalltau(data['wer_rank'], data['pred_rank'])[0])
print('kendall %f' % stats.spearmanr(data['wer_rank'], data['pred_rank'])[0])

print('Correlations with WER score itself')

print('pearson %f' % stats.pearsonr(data['WER, uncased, not punctuated'], -data['preds'])[0])
print('spearman %f' % stats.spearmanr(data['WER, uncased, not punctuated'], -data['preds'])[0])
print('kendall %f' % stats.kendalltau(data['WER, uncased, not punctuated'], -data['preds'])[0])

