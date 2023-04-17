import torch, re, time
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
parser = ArgumentParser(description='Input parameters')
parser.add_argument('--filename', default='en-libre.csv', type=str, help='Enter name of the .csv file filename e.g. en-libre.csv')
parser.add_argument('--modelname', default='self_super.ckpt', type=str, help='Enter name of the model e.g. self_super.ckpt')
args = parser.parse_args()

class Smish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * (x.sigmoid() + 1).log().tanh()

class aiXER(nn.Module):
  def __init__(self, model_name:str, max_length:int=128):
    super().__init__()
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)

    hidden_size = 32

    self.dense = nn.Sequential(nn.Dropout(0.1), nn.Linear(384, hidden_size, bias = False), 
        nn.Dropout(0.1), Smish(), nn.Linear(hidden_size, 1, bias = False))

  def forward(self, x):
    hyps_inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
    h = self.model(**hyps_inputs).pooler_output
    return self.dense(h).sigmoid().squeeze(-1)

class InferenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return sentence

def inference(model, dataloader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for sentences in dataloader:
            outputs.append(model(sentences))
        return torch.cat(outputs)

model = aiXER('nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large')
checkpoint = torch.load(args.modelname, map_location='cpu')
model_weights = checkpoint["state_dict"]
model.load_state_dict(model_weights)
model.eval()

print('Producing the Results for %s\nUse --filename to change the file.' % args.filename)

data = pd.read_csv(args.filename)


data = data[data['WER, uncased, not punctuated'].notna()]

data['WER, uncased, not punctuated'] = data['WER, uncased, not punctuated'].str.strip('%').astype(float)

sentences = data['outputText'].astype(str).to_list()

sentences = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]

inference_dataset = InferenceDataset(sentences)
dataloader = DataLoader(inference_dataset, batch_size=16, shuffle=False)

start = time.time()
data['preds'] = inference(model, dataloader).numpy()
end = time.time()

elapsed = end - start
print(elapsed/len(dataloader))

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
'''
wers = torch.Tensor(data["rank"].values)

from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef, KendallRankCorrCoef

print(PearsonCorrCoef()(wers, preds))
print(SpearmanCorrCoef()(wers, preds))
print(KendallRankCorrCoef()(wers, preds))
'''
