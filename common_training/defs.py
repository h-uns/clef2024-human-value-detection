import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, recall_score
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path

HV_LABELS = ('Self-direction: thought attained',
             'Self-direction: thought constrained',
             'Self-direction: action attained',
             'Self-direction: action constrained',
             'Stimulation attained',
             'Stimulation constrained',
             'Hedonism attained',
             'Hedonism constrained',
             'Achievement attained',
             'Achievement constrained',
             'Power: dominance attained',
             'Power: dominance constrained',
             'Power: resources attained',
             'Power: resources constrained',
             'Face attained',
             'Face constrained',
             'Security: personal attained',
             'Security: personal constrained',
             'Security: societal attained',
             'Security: societal constrained',
             'Tradition attained',
             'Tradition constrained',
             'Conformity: rules attained',
             'Conformity: rules constrained',
             'Conformity: interpersonal attained',
             'Conformity: interpersonal constrained',
             'Humility attained',
             'Humility constrained',
             'Benevolence: caring attained',
             'Benevolence: caring constrained',
             'Benevolence: dependability attained',
             'Benevolence: dependability constrained',
             'Universalism: concern attained',
             'Universalism: concern constrained',
             'Universalism: nature attained',
             'Universalism: nature constrained',
             'Universalism: tolerance attained',
             'Universalism: tolerance constrained')

# 'Humility' is placed at the end because it will be filtered out and the labels should remain contiguous afterwards
HV_CODES = {0:  'No label',
            1:  'Self-direction: thought',
            2:  'Self-direction: action',
            3:  'Stimulation',
            4:  'Hedonism',
            5:  'Achievement',
            6:  'Power: dominance',
            7:  'Power: resources',
            8:  'Face',
            9:  'Security: personal',
            10: 'Security: societal',
            11: 'Tradition',
            12: 'Conformity: rules',
            13: 'Conformity: interpersonal',
            14: 'Universalism: tolerance',
            15: 'Benevolence: caring',
            16: 'Benevolence: dependability',
            17: 'Universalism: concern',
            18: 'Universalism: nature',
            19: 'Humility'}



# Initial training experiments were not able to learn the label 19 (Humility), so it will be taken out of the model 
HV_CODES_FILTERED = {k: v for k,v in HV_CODES.items() if k != 19}


def get_model_dir_name(random_seed, weighted, for_attainment, for_english):
    rs = str(random_seed)
    target = 'attn' if for_attainment else 'hv'
    lang = 'eng' if for_english else 'noneng'
    weight = 'w' if weighted else 'nw'
    
    return f'RS_{rs}_{target}_{lang}_{weight}'
    

def get_hv_label(row):
    # Presumption: The row contains at most one label
    for hv_code, hv_name in HV_CODES.items():
        
        if hv_code == 0: continue # skip 'no label'
        
        if row[f"{hv_name} attained"] != 0 or row[f"{hv_name} constrained"] != 0:
            return hv_code
    
    return 0 # no label

def get_attainment(row):
    
    hv = row['hv_label']
    
    if hv == 0:
        return 2 # unknown
    
    hv_name = HV_CODES.get(hv)
    
    if row[f"{hv_name} constrained"] == 1:
        return 0
    elif row[f"{hv_name} attained"] == 1:
        return 1
    else:
        return 2 # unknown


def create_representative_split(df_original):
    
    from sklearn.model_selection import train_test_split
    
    # the same validation set will be used for all models
    VALIDATION_SPLIT_SEED = 66
    
    train_subsets = []
    validation_subsets = []
    
    langs_vc = df_original['lang'].value_counts()
    
    for lang in langs_vc.index:
        
        lang_df_original = df_original[df_original['lang'] == lang]
        vc_labels_original = lang_df_original['hv_label'].value_counts(normalize=True)
        
        for label in vc_labels_original.index:
            
            label_df_original = lang_df_original[lang_df_original['hv_label'] == label]
            
            sample_size = 0.1
            
            if len(label_df_original) < 10 and len(label_df_original) >= 5:
                sample_size = 0.2
            elif len(label_df_original) < 5:
                print(f'Very few instances of {lang} and {label}')
                train_subsets.append(label_df_original)
                continue
            
            train_subset, validation_subset = train_test_split(label_df_original, test_size=sample_size,
                                                               random_state=VALIDATION_SPLIT_SEED)
            train_subsets.append(train_subset)
            validation_subsets.append(validation_subset)
        
    df_train = pd.concat(train_subsets)
    df_validation = pd.concat(validation_subsets)
    
    return df_train, df_validation


def get_dataframes(ds_dir, for_attainment, for_english):
    
    global HV_CODES
    
    training_labels_original = pd.read_csv(path.join(ds_dir, 'training', 'labels.tsv'),
                                           sep='\t', encoding='utf-8')
    training_sentences_original = pd.read_csv(path.join(ds_dir, 'training', 'sentences.tsv'),
                                              sep='\t', encoding='utf-8')

    validation_labels_original = pd.read_csv(path.join(ds_dir, 'validation', 'labels.tsv'),
                                             sep='\t', encoding='utf-8')
    validation_sentences_original = pd.read_csv(path.join(ds_dir, 'validation', 'sentences.tsv'),
                                                sep='\t', encoding='utf-8')

    training_labels = pd.concat([training_labels_original, validation_labels_original],ignore_index=True)
    training_sentences = pd.concat([training_sentences_original, validation_sentences_original], ignore_index=True)

    df_train = pd.merge(training_sentences, training_labels, how='inner', on=['Text-ID', 'Sentence-ID'])
    df_train.drop_duplicates(subset=['Text'], inplace=True)
    
    # included for analytical purposes, but will not be used in training
    df_test = pd.read_csv(path.join(ds_dir, 'test', 'sentences.tsv'), sep='\t', encoding='utf-8')
    
    for df in (df_train, df_test):
        df['lang'] = df['Text-ID'].apply(lambda s: s[0:2])
        df['num_words'] = df['Text'].apply(lambda s: len(s.split()))
    
    df_train['num_labels'] = df_train[list(HV_LABELS)].sum(axis=1)
    
    # initial filtering
    df_train = df_train[df_train['num_labels'] <= 1.0]
    df_train = df_train[df_train['num_words'] > 2]
    
    df_train['hv_label'] = df_train.apply(get_hv_label, axis=1)
    df_train['attainment'] = df_train.apply(get_attainment, axis=1)
    
    df_train = df_train[df_train['hv_label'].isin(HV_CODES_FILTERED)]
    df_train.reset_index(inplace=True, drop=True)
    
    # creating the validation set will be the same regardless of the target column and used language
    df_train, df_validation = create_representative_split(df_train)
    
    if for_attainment:
        # remove rows with unknown attainment values
        df_train = df_train[df_train['attainment'] != 2]
        df_validation = df_validation[df_validation['attainment'] != 2]
        
    if for_english:
        df_train = df_train[df_train['lang'] == 'EN']
        df_validation = df_validation[df_validation['lang'] == 'EN']
    else:
        df_train = df_train[df_train['lang'] != 'EN']
        df_validation = df_validation[df_validation['lang'] != 'EN']
        
    df_train.reset_index(inplace=True, drop=True)
    df_validation.reset_index(inplace=True, drop=True)
    
    return df_train, df_validation, df_test



def get_tokenized_datasets(df_train, df_validation, df_test, target_column, tokenizer):
    
    ds_train = Dataset.from_pandas(df_train)
    ds_validation = Dataset.from_pandas(df_validation)
    ds_test = Dataset.from_pandas(df_test)
    
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c != 'Text' and c != target_column])
    ds_validation = ds_validation.remove_columns([c for c in ds_validation.column_names if c != 'Text' and c != target_column])
    ds_test = ds_test.remove_columns([c for c in ds_test.column_names if c != 'Text'])
    
    def tokenize_function(batch):
        return tokenizer(batch['Text'], truncation=True)
    
    ds_train = ds_train.map(tokenize_function, batched=True, remove_columns=('Text'))
    ds_validation = ds_validation.map(tokenize_function, batched=True, remove_columns=('Text'))
    ds_test = ds_test.map(tokenize_function, batched=True, remove_columns=('Text'))
    
    ds_train = ds_train.rename_column(target_column, 'label')
    ds_validation = ds_validation.rename_column(target_column, 'label')
    
    return DatasetDict({'train': ds_train, 'validation': ds_validation, 'test':ds_test})

def get_class_weights(df_train, df_validation, target_column):
    df_combined = pd.concat([df_train, df_validation], ignore_index=True)
    
    return compute_class_weight(class_weight="balanced",
                                classes=np.sort(pd.unique(df_combined[target_column])),
                                y=df_combined[target_column].tolist())


def get_optimizer_and_dataloaders(ds, model, tokenizer, batch_size):
    
    LEARNING_RATE = 1e-06
    BASE_WEIGHT_DECAY = 0.0
    CLASSIFIER_WEIGHT_DECAY = 0.01
    
    ds.set_format('torch')
    
    grouped_params = [
        {"params": model.base_model.parameters()},
        {"params": model.classifier.parameters(), 'weight_decay': CLASSIFIER_WEIGHT_DECAY}
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=LEARNING_RATE, weight_decay=BASE_WEIGHT_DECAY)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(ds["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
    eval_dataloader = DataLoader(ds["validation"], batch_size=batch_size, collate_fn=data_collator)
    
    return optimizer, train_dataloader, eval_dataloader

def train_model(model,
                train_dataloader, 
                eval_dataloader, 
                optimizer,
                loss_fn,
                for_attainment,
                device,
                num_epochs,
                output_dir,
                use_grad_checkpointing):

    
    if use_grad_checkpointing:
        model.gradient_checkpointing_enable()
    
    model.to(device)
    
    num_training_steps = num_epochs * len(train_dataloader)
    
    progress_bar = tqdm(range(num_training_steps))
    
    report = {}
    all_logits = {}
    
    for epoch in range(num_epochs):
        
        model.train() # set model training mode
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            batch_output = model(**batch)
            # loss = batch_output.loss
            loss = loss_fn(batch_output.logits, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        
        # evaluation
        print('Evaluating...')
        
        model.eval()
        num_evaluation_steps = len(eval_dataloader)
        logit_lists = []
        true_labels = []
        running_loss = 0.0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels')
                batch_output = model(**batch)
                loss = loss_fn(batch_output.logits, labels)
                running_loss += loss.item()
                logit_lists.extend(batch_output.logits.tolist())
                true_labels.extend(labels.tolist())
        
        validation_loss = running_loss / num_evaluation_steps
        
        preds = []
        
        for logit_list in logit_lists:
            logit_tensor = torch.atleast_2d(torch.Tensor(logit_list))
            probs = torch.softmax(logit_tensor, dim=1) 
            preds.append(probs.argmax(dim=1).item())
        
        metrics, metrics_summary = get_metrics(preds, true_labels, validation_loss, for_attainment)
        print(metrics_summary)
        
        report[epoch] = metrics
        all_logits[epoch] = logit_lists
        
        # All checkpoints are saved. Unneeded checkpoints can be manually deleted
        model.save_pretrained(path.join(output_dir, f'ckpt_{epoch}'))
    
    
    if use_grad_checkpointing:
        model.gradient_checkpointing_disable()
    
    return report, all_logits



def get_metrics(preds, true_labels, validation_loss, for_attainment):
    
    metrics = {}
    metrics['validation_loss'] = validation_loss
    
    # label-wise predictions
    binary_preds = {}
    binary_true_labels = {}
    
    available_labels = (0, 1) if for_attainment else HV_CODES_FILTERED.keys()
    
    for l in available_labels:
        binary_preds = list(map(lambda x: 1 if x==l else 0, preds))
        binary_true_labels = list(map(lambda x: 1 if x==l else 0, true_labels))
        metrics[f'l_{l}_accuracy'] = accuracy_score(binary_true_labels, binary_preds)
        metrics[f'l_{l}_recall'] = accuracy_score(binary_true_labels, binary_preds)
        metrics[f'l_{l}_f1'] = f1_score(binary_true_labels, binary_preds)
    
    metrics['f1_macro'] = np.mean([metrics[f'l_{l}_f1'] for l in available_labels])
    
    if not for_attainment:
        metrics['f1_macro_adjusted'] = np.mean([metrics[f'l_{l}_f1'] for l in available_labels if l != 0])
    
    metrics['f1_sklearn'] = f1_score(true_labels, preds, average='macro') # should be the same f1_macro, included only for validation
    metrics['accuracy'] = accuracy_score(true_labels, preds)
    metrics['recall'] = recall_score(true_labels, preds, average='macro')
    
    metrics_summary = 'loss: ' + str(round(metrics['validation_loss'], 5))
    metrics_summary += ' acc: ' + str(round(metrics['accuracy'], 3))
    metrics_summary += ' recall: ' + str(round(metrics['recall'], 3))
    metrics_summary += ' f1_sk: ' + str(round(metrics['f1_sklearn'], 3))
    metrics_summary += ' f1: ' + str(round(metrics['f1_macro'], 3))
    
    if not for_attainment:
        metrics_summary += ' f1_adj: ' + str(round(metrics['f1_macro_adjusted'], 3))

    metrics_summary += '\n'
    for l in available_labels:
        metrics_summary += ' ' + str(round(metrics[f'l_{l}_f1'], 2))
    
    return metrics, metrics_summary