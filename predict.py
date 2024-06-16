import argparse
from os import path
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import Dataset
import pandas as pd
from transformers import (DataCollatorWithPadding, AutoTokenizer,
                          AutoModelForSequenceClassification)

from common_training import (HV_LABELS,
                             HV_CODES_FILTERED)



CONSTRAINED_COLUMNS = tuple(l for l in HV_LABELS if 'constrained' in l)
ATTAINED_COLUMNS = tuple(l for l in HV_LABELS if 'attained' in l)

HV_ENSEMBLE_ENG = ('RS_66_hv_eng_nw',
                'RS_66_hv_eng_w',
                'RS_67_hv_eng_nw',
                'RS_67_hv_eng_w',
                )

HV_ENSEMBLE_NONENG = ('RS_66_hv_noneng_nw',
                   'RS_67_hv_noneng_nw',
                   'RS_66_hv_noneng_w',
                   'RS_67_hv_noneng_w',
                   )

ATTN_MODEL_ENG = 'RS_66_attn_eng_nw'
ATTN_MODEL_NONENG = 'RS_66_attn_noneng_nw'


def create_run_file(sentences_file, output_file, hv_ensemble_eng, hv_ensemble_noneng,
                    attn_model_eng, attn_model_noneng):
    
    df_test = pd.read_csv(sentences_file, sep='\t', encoding='utf-8')
    df_test['lang'] = df_test['Text-ID'].apply(lambda s: s[0:2])
    df_test['num_words'] = df_test['Text'].apply(lambda s: len(s.split()))
    
    df_eng = df_test[df_test['lang'] == 'EN'].copy()
    df_noneng = df_test[df_test['lang'] != 'EN'].copy()
    
    tokenizer_eng = AutoTokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')
    tokenizer_noneng = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')
    
    
    dataloader_eng = get_run_dataloader(df_eng, tokenizer_eng)
    dataloader_noneng = get_run_dataloader(df_noneng, tokenizer_noneng)
    
    fast_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    slow_device = torch.device("cpu")
    threshold_eng = 0.44
    threshold_noneng = 0.49
    
    avg_hv_preds_eng = get_avg_preds_from_models(hv_ensemble_eng,
                                          dataloader_eng,
                                          fast_device,
                                          slow_device,
                                          threshold_eng)
    
    avg_hv_preds_noneng = get_avg_preds_from_models(hv_ensemble_noneng,
                                             dataloader_noneng,
                                             fast_device,
                                             slow_device,
                                             threshold_noneng)
    
    
    attainment_preds_eng = get_attainent_preds(attn_model_eng, dataloader_eng, fast_device, slow_device)
    attainment_preds_noneng = get_attainent_preds(attn_model_noneng, dataloader_noneng, fast_device, slow_device)

    df_eng['hv_pred'] = avg_hv_preds_eng
    df_noneng['hv_pred'] = avg_hv_preds_noneng
    
    df_eng['attn_pred'] = attainment_preds_eng
    df_noneng['attn_pred'] = attainment_preds_noneng
    
    df_run = df_test[['Text-ID', 'Sentence-ID', 'num_words']].copy()
    
    for l in HV_LABELS:
        df_run[l] = 0.0
    df_run.reset_index(inplace=True, drop=True)
    
    for df in (df_eng, df_noneng):
        for i in df.index:
            row = df.loc[i]
            t_id = row['Text-ID']
            s_id = row['Sentence-ID']
            pred = row['hv_pred']
            n_words = row['num_words']
            attainment = row['attn_pred']
            
            if n_words <= 2: # hv-predictions are not reliable for short sentences
                pred = 0
            
            assert attainment == 1 or attainment == 0
            is_attained = attainment == 1
            
            run_row = df_run[(df_run['Text-ID'] == t_id) & (df_run['Sentence-ID'] == s_id)].copy()
            assert len(run_row) == 1
            
            adjust_run_row(run_row, pred, is_attained)
            df_run.loc[(df_run['Text-ID'] == t_id) & (df_run['Sentence-ID'] == s_id)] = run_row
    
    df_run.drop('num_words', axis=1, inplace=True)
    
    df_run.to_csv(output_file, index=False, sep='\t', encoding='utf-8')
    
    
def get_run_dataloader(df, tokenizer):
    
    def tokenize_function(batch):
        return tokenizer(batch['Text'], truncation=True)
    
    ds = Dataset.from_pandas(df)
    ds = ds.remove_columns([c for c in ds.column_names if c != 'Text'])
    ds = ds.map(tokenize_function, batched=True, remove_columns=('Text',))
    ds.set_format('torch')
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return DataLoader(ds, batch_size=32, collate_fn=data_collator)

def adjust_run_row(row, hv_label, is_attained):
    
    if is_attained:
        for c in CONSTRAINED_COLUMNS:
            row[c] = 0.1
        for c in ATTAINED_COLUMNS:
            row[c] = 0.2
    else:
        for c in CONSTRAINED_COLUMNS:
            row[c] = 0.2
        for c in ATTAINED_COLUMNS:
            row[c] = 0.1
    
    if hv_label != 0:
        postfix = ' attained' if is_attained else ' constrained'
        column_name = HV_CODES_FILTERED[hv_label] + postfix
        row[column_name] = 0.6
    

def get_attainent_preds(model_path, dataloader, fast_device, slow_device):
    
    logit_lists = get_logits(model_path, dataloader, fast_device, slow_device)
    preds = get_predictions(logit_lists)
    return  [x[0] for x in preds]


def get_avg_preds_from_models(model_paths, dataloader, fast_device, slow_device, threshold):
    
    preds = get_preds_from_models(model_paths, dataloader, fast_device, slow_device)
    
    avg_pred_prob_pairs = get_avg_preds(preds, threshold=threshold)
    final_preds = [x[0] for x in avg_pred_prob_pairs]
        
    return final_preds


def get_preds_from_models(model_paths, dataloader, fast_device, slow_device):    
    preds = {}
    
    for model_path in model_paths:
        print(f'Running model {model_path}...')
        logit_lists = get_logits(model_path, dataloader, fast_device, slow_device)        
        preds[model_path] = get_predictions(logit_lists)
        
    return preds

def get_logits(model_path, dataloader, fast_device, slow_device):
    
    logit_lists = [] # list of lists
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(fast_device)
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            if 'labels' in batch.keys():
                batch.pop('labels')
            batch = {k: v.to(fast_device) for k, v in batch.items()}
            logit_lists.extend(model(**batch).logits.tolist())
        
    model.to(slow_device)
    del(model)

    return logit_lists


def get_predictions(logit_lists):
    
    pred_prob_pairs = []
    
    for logit_list in logit_lists:
        logit_tensor = torch.atleast_2d(torch.Tensor(logit_list))
        probs = torch.softmax(logit_tensor, dim=1)
        max_elem = probs.max(dim=1)
        
        pred = max_elem.indices.item()
        prob = max_elem.values.item()
        
        pred_prob_pairs.append((pred, prob))
    
    return pred_prob_pairs


def get_avg_preds(model_outputs, threshold):
    
    # all outputs have the same number of predictions
    num_instances = len(model_outputs[list(model_outputs.keys())[0]]) 
    avg_pred_prob_pairs = []
    
    for i in range(0, num_instances):
        
        # key: prediction, value: list of probabilities
        pred_prob_for_instance = {}
        
        for model_output in model_outputs.values():
            pred, prob = model_output[i]
            if pred in pred_prob_for_instance.keys():
                pred_prob_for_instance[pred].append(prob)
            else:
                pred_prob_for_instance[pred] = [prob]
        
        pred_prob_for_instance = get_safe_preds(pred_prob_for_instance, threshold)
        
        aggregated_preds_for_instance = {}
        
        for k, v in pred_prob_for_instance.items():
            aggregated_preds_for_instance[k] = np.mean(v)
        
        
        avg_pred = max(aggregated_preds_for_instance, key=aggregated_preds_for_instance.get)
        avg_pred_prob = aggregated_preds_for_instance[avg_pred]
        avg_pred_prob_pairs.append((avg_pred, avg_pred_prob))
    
    return avg_pred_prob_pairs

def get_safe_preds(pred_prob_list, threshold):
    
    safe_pred_prob_list = {}
    
    for pred in pred_prob_list.keys():
        safe_probs = []
        
        for prob in pred_prob_list[pred]:
            if prob >= threshold:
                safe_probs.append(prob)
        
        if len(safe_probs) > 0:
            safe_pred_prob_list[pred] = safe_probs
    
    if len(safe_pred_prob_list) > 0:
        return safe_pred_prob_list # return the original if there are no safe predictions
    else:
        return pred_prob_list




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-file', dest='sentences_file', type=str, required=True,
                        help='Path of the input .tsv file')
    parser.add_argument('--output-file', dest='output_file', type=str, required=True,
                        help='Path of the output run file')
    parser.add_argument('--models-dir', dest='models_dir', type=str, required=True,
                        help='Directory that contains the fine-tuned models')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    hv_ensemble_eng = map(lambda model_name: path.join(args.models_dir, model_name), HV_ENSEMBLE_ENG)
    hv_ensemble_noneng = map(lambda model_name: path.join(args.models_dir, model_name), HV_ENSEMBLE_NONENG)
    attn_model_eng = path.join(args.models_dir, ATTN_MODEL_ENG)
    attn_model_noneng = path.join(args.models_dir, ATTN_MODEL_NONENG)
    
    create_run_file(args.sentences_file, args.output_file,
                    hv_ensemble_eng, hv_ensemble_noneng,
                    attn_model_eng, attn_model_noneng)