import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import grad

class Smish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * (x.sigmoid() + 1).log().tanh()


# Define the inference dataset
class InferenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return sentence


def inference_attention(model, dataloader, return_attentions=False):
    model.eval()
    outputs = []
    first_attentions = []
    last_attentions = []
    avg_all_attentions = []
    with torch.no_grad():
        for sentences in dataloader:
            if return_attentions:
                output, first_attention, last_attention, avg_all_attention = model(sentences, return_attentions=True)
                outputs.append(output)
                first_attentions.append(first_attention)
                last_attentions.append(last_attention)
                avg_all_attentions.append(avg_all_attention)
            else:
                output = model(sentences)
                outputs.append(output)
        if return_attentions:
            return torch.cat(outputs), first_attentions, last_attentions, avg_all_attentions
        else:
            return torch.cat(outputs)


class aiXER_norm_attention_value(nn.Module):
    def __init__(self, model_name:str, max_length:int=128):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        hidden_size = 32

        self.dense = nn.Sequential(nn.Dropout(0.1), nn.Linear(384, hidden_size, bias = False), 
            nn.Dropout(0.1), Smish(), nn.Linear(hidden_size, 1, bias = False))

    def forward(self, x, return_attentions=False):
        hyps_inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        model_outputs = self.model(**hyps_inputs, output_attentions=return_attentions)
        h = model_outputs.pooler_output
        dense_output = self.dense(h).sigmoid().squeeze(-1)
        
        if return_attentions:
            # Extract attention and value vectors
            attention_outputs = model_outputs.attentions  
            scaled_attentions = []
            for attention_value, layer in zip(attention_outputs,self.model.encoder.layer) :
                values = layer.attention.self.value(model_outputs.last_hidden_state)
                value_norms = torch.norm(values, dim=-1)

                attention_weights = attention_value
                scaled_attention = attention_weights / value_norms.unsqueeze(1).unsqueeze(-1) 
                scaled_attentions.append(scaled_attention)
            
            first_layer_attentions = scaled_attentions[0]
            last_layer_attentions = scaled_attentions[-1]
            avg_attentions = torch.mean(torch.stack(scaled_attentions), dim=0)

            return dense_output, first_layer_attentions, last_layer_attentions, avg_attentions
        else:
            return dense_output
        

