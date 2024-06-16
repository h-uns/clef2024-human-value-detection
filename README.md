# Training the models

Example:

```
python train.py \
       --num-epochs 10 \
       --random-seed 66 \
       --loss-fn-type non-weighted \
       --target-column hv_label \
       --model-langs non-english \
       --ds-dir /path/to/dataset \
       --ckpt-dir ./checkpoints
```

> **Note:** All the training checkponts will be saved in `ckpt-dir`. Un-needed checkpoints have to be manually deleted.

# Reproducing the submission file

```
# takes around 15 minutes with V100S
python predict.py \
       --sentences-file /path/to/test/sentences.tsv \
       --output-file ./run.tsv \
       --models-dir /path/to/finetuned_models
```



