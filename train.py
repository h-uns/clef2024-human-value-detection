import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
from os import path
from common_training import (HV_CODES_FILTERED,
                             get_model_dir_name,
                             get_dataframes,
                             get_tokenized_datasets,
                             get_class_weights,
                             get_optimizer_and_dataloaders,
                             train_model)



def main(num_epochs,random_seed, loss_fn_type, target_column, model_langs, ds_dir, ckpt_dir):
    
    
    # Derived configurations
    use_weighted_loss = (loss_fn_type == 'weighted')
    for_english = (model_langs == 'english')
    for_attainment = (target_column == 'attainment')
    
    if for_attainment:
        num_labels = 2
    else:
        num_labels = len(HV_CODES_FILTERED)

    if for_english:
        model_name = 'microsoft/deberta-v2-xxlarge'
        # xxlarge model requires gradient checkpointing and a smaller batch size (to save memory)
        use_grad_checkpointing = True
        batch_size = 8
    else:
        model_name = 'FacebookAI/xlm-roberta-large'
        use_grad_checkpointing = False
        batch_size = 16
        
    
    model_dir_name = get_model_dir_name(random_seed=random_seed,
                                        weighted=use_weighted_loss,
                                        for_attainment=for_attainment,
                                        for_english=for_english)
    output_dir = path.join(ckpt_dir, model_dir_name)

    transformers.set_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train, df_validation, df_test = get_dataframes(ds_dir=ds_dir,
                                                      for_attainment=for_attainment,
                                                      for_english=for_english)


    ds = get_tokenized_datasets(df_train, df_validation, df_test, target_column, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if use_weighted_loss:
        class_weights = torch.Tensor(get_class_weights(df_train, df_validation, target_column)).to(torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer, train_dataloader, eval_dataloader = get_optimizer_and_dataloaders(ds, model, tokenizer, batch_size)

    report, all_logits = train_model(model=model,
                                     train_dataloader=train_dataloader,
                                     eval_dataloader=eval_dataloader,
                                     optimizer=optimizer,
                                     loss_fn=loss_fn,
                                     for_attainment=for_attainment,
                                     device=device,
                                     num_epochs=num_epochs,
                                     output_dir=output_dir,
                                     use_grad_checkpointing=use_grad_checkpointing)
    
    
    # Save report and logits for offline analysis
    filepath_report = path.join(ckpt_dir, 'report.pkl')
    filepath_logits = path.join(ckpt_dir, 'logits.pkl')

    with open(filepath_report, 'wb') as outp:
        pickle.dump(report, outp)

    with open(filepath_logits, 'wb') as outp:
        pickle.dump(all_logits, outp)
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', dest='num_epochs',  type=int, required=True,
                        help='Number of training epochs')
    parser.add_argument('--random-seed', dest='random_seed', type=int, required=True,
                        help='Random seed to be used for fine-tuning the model')
    parser.add_argument('--loss-fn-type', dest='loss_fn_type', choices=['weighted', 'non-weighted'], required=True,
                        help='Possible values: "weighted, "non-weighted"')
    parser.add_argument('--target-column', dest='target_column', choices=['attainment', 'hv_label'], required=True,
                        help='Possible values: "attainment, "hv_label"')
    parser.add_argument('--model-langs', dest='model_langs', choices=['english', 'non-english'], required=True,
                        help='Possible values: "english, "non-english"')
    parser.add_argument('--ds-dir', dest='ds_dir', type=str, required=True,
                        help='Dataset home directory')
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, required=True,
                        help='Directory where the training checkpoints will be saved')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    main(num_epochs=args.num_epochs,
         random_seed=args.random_seed,
         loss_fn_type=args.loss_fn_type,
         target_column=args.target_column,
         model_langs=args.model_langs,
         ds_dir=args.ds_dir,
         ckpt_dir=args.ckpt_dir)