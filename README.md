# Team arthur-schopenhauer for ValueEval'24

## Local execution

### Training the models

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

### Reproducing the submission file

```
# takes around 15 minutes with V100S
python predict.py \
       --sentences-file /path/to/test/sentences.tsv \
       --output-file ./run.tsv \
       --models-dir /path/to/finetuned_models
```

## Dockerization

### Training
```bash
# build
docker build -f Dockerfile_train -t valueeval24-arthur-schopenhauer-train-ensemble:1.0.0 .

# run
docker run --rm \
  -v "$PWD/valueeval24:/dataset" -v "$PWD/models:/models" \
  valueeval24-arthur-schopenhauer-train-ensemble:1.0.0
```

### Prediction
- uses models from [https://huggingface.co/h-uns](https://huggingface.co/h-uns)
```bash
# build
docker build -f Dockerfile_predict -t valueeval24-arthur-schopenhauer-ensemble:1.0.0 .

# run
docker run --rm \
  -v "$PWD/valueeval24/test:/dataset" -v "$PWD/output:/output" \
  valueeval24-arthur-schopenhauer-ensemble:1.0.0

# view results
cat output/run.tsv
```

