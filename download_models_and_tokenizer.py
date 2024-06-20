from os import path
from transformers import (AutoConfig, AutoTokenizer,
                          AutoModelForSequenceClassification)


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


def download_tokenizer(tokenizer_dir: str = "/tokenizer"):
    for model_name in ['microsoft/deberta-v2-xxlarge', 'FacebookAI/xlm-roberta-large']:
        print(f"Downloading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        tokenizer.save_pretrained(path.join(tokenizer_dir, model_name))
        config.save_pretrained(path.join(tokenizer_dir, model_name))


def download_models(models_dir: str = "/models"):
    for model_name in list(HV_ENSEMBLE_ENG) + list(HV_ENSEMBLE_NONENG) + [ATTN_MODEL_ENG, ATTN_MODEL_NONENG]:
        model_path = path.join('h-uns', model_name)
        print(f"Downloading model from huggingface: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.save_pretrained(path.join(models_dir, model_name))


if __name__ == "__main__":
    download_models()
    download_tokenizer()

