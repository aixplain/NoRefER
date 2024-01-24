from norefer import *

import numpy as np
import ast
import jiwer
from scipy.stats import spearmanr
from itertools import zip_longest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


def process_attention_values(attention_values):
    '''
    Assuming attention_values is a list of tensors with shape [batch_size, num_heads, seq_len, seq_len]:
    1. Average across attention heads - Shape: [batch_size, seq_len, seq_len]
    2. Average across all tokens to get a single score per token -  Shape: [seq_len]
    '''    
    token_attention_scores = []

    for batch_attention in attention_values:
        # Average across attention heads
        avg_attention = batch_attention.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]

        for sentence_attention in avg_attention:
            # Average across all tokens to get a single score per token
            token_scores = sentence_attention.mean(dim=1).tolist()  # Shape: [seq_len]
            token_attention_scores.append(token_scores)

    return token_attention_scores


def tokenize_sentence(sentences, tokenizer):
    """
    Tokenizes a sentence using the provided tokenizer from the transformers library.

    :param sentence: The sentence to tokenize.
    :param tokenizer: The tokenizer used by the model.
    :return: A list of token strings.
    """
    tokenized_sentences = []

    for sentence in sentences:
        # Tokenize the sentence and get the input IDs
        input_ids = tokenizer.encode(sentence, add_special_tokens=True)

        # Convert the input IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tokenized_sentences.append(tokens)

    return tokenized_sentences


def process_transcription_attention(transcription, need_split=True):
    # Initialize the model and tokenizer here
    model_name = 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large'
    model = aiXER_norm_attention_value(model_name)
    model_path = "../checkpoints/semi_super.ckpt"  # Adjust the path as necessary
    checkpoint = torch.load(model_path, map_location='cpu')
    model_weights = checkpoint["state_dict"]
    model.load_state_dict(model_weights, strict=False)
    model.eval() 
    if need_split:
        # Split the transcription into sentences
        sentences = transcription.split(',')
        # Clean the sentences
        sentences = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]
    else:
        sentences = transcription
    
    # Get the token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenize_sentence(sentences, tokenizer)

    # Create a DataLoader
    inference_dataset = InferenceDataset(sentences)
    dataloader = DataLoader(inference_dataset, batch_size=16, shuffle=False)

    # Perform inference
    scores, _, _, avg_all_attention_values = inference_attention(model, dataloader, return_attentions=True)
    norm_avg_attentions = process_attention_values(avg_all_attention_values)

    # Create a DataFrame
    data = pd.DataFrame({
        'outputText': sentences,
        'outputToken': tokens,
        'preds': scores,
        'norm_avg_attentions': norm_avg_attentions,
    })

    return data


def aggregate_scores(scores, method):
    """
    Aggregate a list of scores based on the specified method.

    :param scores: List of scores to aggregate.
    :param method: Aggregation method ('average', 'max', or 'q3').
    :return: Aggregated score.
    """
    if method == 'average':
        return sum(scores) / len(scores)
    elif method == 'max':
        return max(scores)
    elif method == 'q3':
        return np.percentile(scores, 75)
    else:
        raise ValueError("Invalid aggregation method. Choose 'average', 'max', or 'q3'.")
    

def calculate_word_scores_with_tokens(df, attention_col, aggregation_method='average'):
    """
    Calculate the score for each word in a sentence by aggregating the scores of its tokens.
    Aggregation methods can be 'average', 'max', or 'q3' (third quartile).

    :param df: DataFrame containing sentences and token attentions.
    :param attention_col: Column name containing attention values for tokens.
    :param aggregation_method: Method to aggregate token scores ('average', 'max', or 'q3').
    :return: DataFrame with an additional column for word attentions and words.
    """
    word_attentions_dict = []
    word_attentions_list = []
    words_list = []

    for index, row in df.iterrows():
        token_attentions = row[attention_col]
        tokens = row['outputToken']

        word_attention_map = {}
        current_word = ""
        current_scores = []
        word_attentions = []
        words = []
        first_token = True

        for token, score in zip(tokens, token_attentions):
            if token.startswith("▁") or token == "<s>":
                if not first_token:
                    word_attention_map[current_word] = aggregate_scores(current_scores, aggregation_method)
                    word_attentions.append(aggregate_scores(current_scores, aggregation_method))
                    words.append(current_word)

                current_word = token[1:] if token != "<s>" else token
                current_scores = [score]
                first_token = False
            else:
                current_word += token
                current_scores.append(score)

        if current_word and current_scores:
            word_attention_map[current_word] = aggregate_scores(current_scores, aggregation_method)
            word_attentions.append(aggregate_scores(current_scores, aggregation_method))
            words.append(current_word)

        if len(words) > 1 and words[0] == "<s>":
            words[0] += words[1]
            words.pop(1)

            if len(word_attentions) > 1:
                word_attentions[0] = (word_attentions[0] + word_attentions[1]) / 2
                word_attentions.pop(1)

        word_attentions_dict.append(word_attention_map)
        word_attentions_list.append(word_attentions)
        words_list.append(words)

    df['word_attentions_dict'] = word_attentions_dict
    df['word_attentions'] = word_attentions_list
    df['words'] = words_list
    return df


def calculate_word_scores_with_tokens_grad(df, attention_col, aggregation_method='average'):
    """
    Calculate the score for each word in a sentence by aggregating the scores of its tokens.
    Aggregation methods can be 'average', 'max', or 'q3' (third quartile).

    :param df: DataFrame containing sentences and token attentions.
    :param attention_col: Column name containing attention values for tokens.
    :param aggregation_method: Method to aggregate token scores ('average', 'max', or 'q3').
    :return: DataFrame with an additional column for word attentions and words.
    """
    word_attentions_dict = []
    word_attentions_list = []
    words_list = []

    for index, row in df.iterrows():
        token_attentions = row[attention_col]
        tokens = row['outputToken']

        word_attention_map = {}
        current_word = ""
        current_scores = []
        word_attentions = []
        words = []
        first_token = True

        for token, score in zip(tokens, token_attentions):
            if token.startswith("▁") or token == "<s>":
                if not first_token:
                    word_attention_map[current_word] = aggregate_scores(current_scores, aggregation_method)
                    word_attentions.append(aggregate_scores(current_scores, aggregation_method))
                    words.append(current_word)

                current_word = token[1:] if token != "<s>" else token
                current_scores = [score]
                first_token = False
            else:
                current_word += token
                current_scores.append(score)

        if current_word and current_scores:
            word_attention_map[current_word] = aggregate_scores(current_scores, aggregation_method)
            word_attentions.append(aggregate_scores(current_scores, aggregation_method))
            words.append(current_word)

        if len(words) > 1 and words[0] == "<s>":
            words[0] += words[1]
            words.pop(1)

            if len(word_attentions) > 1:
                word_attentions[0] = (word_attentions[0] + word_attentions[1]) / 2
                word_attentions.pop(1)

        word_attentions_dict.append(word_attention_map)
        word_attentions_list.append(word_attentions)
        words_list.append(words)

    df['word_grad_dict'] = word_attentions_dict
    df['word_grad'] = word_attentions_list
    df['words'] = words_list
    return df


def expand_and_rank_words(df, words_col, attentions_col, jiwer_col):
    # Create an empty DataFrame for the expanded information
    expanded_data = []
    faulty_data = []
    # Iterate through each row in the original DataFrame
    for index, row in df.iterrows():
        try:
            words = ast.literal_eval(row[words_col])
            attentions = ast.literal_eval(row[attentions_col])
            jiwers = ast.literal_eval(row[jiwer_col])

            # Combine words and attentions into a list of tuples and sort them by attention values
            word_attention_pairs = sorted(zip(words, attentions, jiwers), key=lambda pair: pair[1])

            # Extend the expanded data with this data, including the original row index for reference
            for word, attention, jiwer_score in word_attention_pairs:
                expanded_data.append({
                    'Word': word,
                    'Attention': attention,
                    'jiwer': jiwer_score,
                    'Original_Index': index
                })
        except:
            faulty_data.append({
                    'Word': word,
                    'Attention': attention,
                    'Original_Index': index
                })

    # Create a new DataFrame from the expanded data
    expanded_df = pd.DataFrame(expanded_data)

    # Sort the expanded DataFrame based on attention scores
    expanded_df.sort_values('Attention', ascending=True, inplace=True)

    return expanded_df


def get_word_fault_scores_jiwer(reference_sentences, hypothesis_sentences):
    combined_word_scores = []

    for reference_sentence, hypothesis_sentence in zip(reference_sentences, hypothesis_sentences):
        # Convert both sentences to lowercase
        try:
            reference_sentence_lower = reference_sentence.lower()
            hypothesis_sentence_lower = hypothesis_sentence.lower()
        except:
            print(reference_sentence)
            print(hypothesis_sentence)

        # Process the sentences to get the alignment
        alignment_output = jiwer.process_words(reference_sentence_lower, hypothesis_sentence_lower)

        # Initialize a list to store the words and scores (including insertions)
        word_scores = []

        # Current index in the reference words
        ref_idx = 0
        reference_words = reference_sentence_lower.split()

        # Process the alignment to determine the scores
        for alignment_chunk in alignment_output.alignments[0]:
            if alignment_chunk.type == 'equal':
                # Add corrected word with score 0
                for idx in range(ref_idx, alignment_chunk.ref_end_idx):
                    word_scores.append((reference_words[idx], 0))
                ref_idx = alignment_chunk.ref_end_idx

            elif alignment_chunk.type == 'substitute':
                # Add substituted word with score 1
                for idx in range(ref_idx, alignment_chunk.ref_end_idx):
                    word_scores.append((reference_words[idx], 1))
                ref_idx = alignment_chunk.ref_end_idx

            elif alignment_chunk.type == 'delete':
                # Add deleted word with score 2
                for idx in range(ref_idx, alignment_chunk.ref_end_idx):
                    word_scores.append((reference_words[idx], 2))
                ref_idx = alignment_chunk.ref_end_idx

            elif alignment_chunk.type == 'insert':
                # Add "inserted" with score 3
                for idx in range(ref_idx, alignment_chunk.hyp_end_idx):
                    word_scores.append(("inserted", 3))
                ref_idx = alignment_chunk.ref_end_idx

        # Append remaining correct words
        for idx in range(ref_idx, len(reference_words)):
            word_scores.append((reference_words[idx], 0))

        # Append the results for this sentence pair to the combined list
        combined_word_scores.append(word_scores)

    return combined_word_scores


def align_attention_with_jiwer(all_word_jiwer_scores, all_word_attentions):
    all_aligned_attentions = []

    for word_jiwer_scores, word_attentions in zip(all_word_jiwer_scores, all_word_attentions):
        aligned_attentions = []

        for jiwer_score, attention in zip_longest(word_jiwer_scores, word_attentions, fillvalue ='None'):
            # Check if jiwer_score is not deletion or insertion
            if jiwer_score not in [2]:
                aligned_attentions.append(attention)
            else:
                att = 0
                aligned_attentions.append(att)

        all_aligned_attentions.append(aligned_attentions)

    return all_aligned_attentions


def process_transcription_gradient(transcription):
    tokenizer = AutoTokenizer.from_pretrained("aixplain/NoRefER")
    model = AutoModel.from_pretrained("aixplain/NoRefER", trust_remote_code=True)
    token_gradients =[]
    for sentence in transcription:
        positive_inputs = tokenizer(sentence, padding=True, return_tensors="pt")
        outputs = model.roberta(**positive_inputs)
        loss = model.dense(outputs.pooler_output).sigmoid().squeeze(-1).mean()  # Take the mean for a scalar loss
        # loss = model.dense(outputs.pooler_output).squeeze(-1).mean() 
        grads = torch.autograd.grad(loss, outputs.last_hidden_state, retain_graph=True, grad_outputs=torch.ones_like(loss))[0]
        token_gradient = grads.sum(dim=1).tolist()
        token_gradients.append(token_gradient[0])
    return token_gradients
