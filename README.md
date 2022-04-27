This repo contains codes for the following paper:

<it>Seo Yeon Park, Cornelia Cargaea</it> : A Data Cartography based MixUp for Pre-trained Language Models (NAACL 2022)

If you would like to refer to it, please cite the paper mentioned above.

## Getting Started
In order to run the code, you have to prepare top 33% easy-to-learn and top 33% ambiguous samples using [this repository](https://github.com/allenai/cartography). We use dataset released by [https://github.com/shreydesai/calibration](https://drive.google.com/file/d/1ro3Q7019AtGYSG76KeZQSq35lBi7lU3v/view). Note that our implementation is based on the implementation provided by [this repository](https://github.com/shreydesai/calibration)


## Requirements
Configure the environments using below command. Our experiments are done by using python 3.6:

```
pip install -r requirements.txt
```


## Characterized Data Samples using Training Dynamics
In our proposed MixUp, we first need to apply AUMs on the top 33% easy-to-learn and the top 33% ambiguous samples to filter possibly mis-labeled. Below we provide an example script for applying AUM on SNLI data, to filter out possibly mis-labeled instances on top 33% easy-to-learn samples. Models were trained using a single NVIDIA RTX A6000 48G GPU.

```
export DEVICE=0
export MODEL="bert-base-uncased"  
export TASK="SNLI"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=256

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi


python3 train.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/${TASK}_${MODEL}_easy_threshold.pt" \
    --output_path "output/${TASK}_${MODEL}_easy_threshold.json" \
    --train_path "calibration_data/${TASK}/train_easy33_bert.tsv" \
    --dev_path "calibration_data/${TASK}/dev.tsv" \
    --test_path "calibration_data/${TASK}/test.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_evaluate \
    --do_train \
    --threshold_sample

```

After you finish fine-tuning BERT measuring AUMs on a subset of training samples (by inserting fake data which are samples reassigned their labels to classes that are different from the original labels including non-existing class), we filter out possibly mis-labeled samples in the most easy-to-learn instances by executing the following scripts. 


```
python3 refine_dataset_using_threshold_aum.py \
    --task SNLI \
    --data_type easy #options: ambig, easy \
    --model_type bert-base-uncased \
    --data_path "calibration_data/${TASK}/train_easy33_bert.tsv" \
    --sampling_ratio 0.8     
```

After these process, you will get train_easy33_bert_woMislabeled.tsv file in the calibration_data folder. In our method, we use sampling_ratio as 0.8 for SNLI, 0.8 for QQP, and 0.5 for SWAG.


## MixUp
Then, we conduct an MixUp in between the most easy-to-learn applied with AUM and the most ambiguous samples by using following scripts. 

```
export DEVICE=0
export MODEL="bert-base-uncased"  
export TASK="SNLI"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=256

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

python3 train.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/${TASK}_${MODEL}_easyWoMislabeled_ambig.pt" \
    --output_path "output/${TASK}_${MODEL}_easyWoMislabeled_ambig.json" \
    --ambig_train_path "calibration_data/${TASK}/train_ambig33_bert.tsv" \
    --easy_train_path "calibration_data/${TASK}/train_easy33_bert_woMislabeled.tsv" \
    --test_path "calibration_data/${TASK}/test.tsv" \
    --dev_path "calibration_data/${TASK}/dev.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_evaluate \
    --do_train \
    --mixup \
    --mixup_type easy_ambig 

```

## Evaluating on in-, out-of-domain test sets

To evaluate fine-tuned model on out-of-domain test set, execute below scripts
```
export DEVICE=0
export MODEL="bert-base-uncased"  
export TASK="MNLI"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=256

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

python3 train.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/SNLI_${MODEL}_easyWoMislabeled_ambig.pt" \
    --output_path "output/${TASK}_${MODEL}_easyWoMislabeled_ambig.json" \
    --test_path "calibration_data/${TASK}/test.txt" \
    --batch_size $BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_evaluate 
```

Then, we evaluate model performance (accuracy) and calibration (ECEs) using the output files dumped in the previous step. 

```
export TEST_PATH="./output/SNLI_bert-base-uncased_easyWoMislabeled_ambig.json"

python3 calibrate.py \
	--test_path $TEST_PATH \
	--do_evaluate
```

## Ablation Study
To conduct an ablation study, we run MixUp on 66\% train set (i.e., conduct MixUp operation between randomly selected samples on 66% train set, which is the union of the top 33% easy-to-learn and the top 33\% ambiguous samples). To do this, execute following scripts and compare the results with our proposed method. 


```
export DEVICE=0
export MODEL="bert-base-uncased"  
export TASK="SNLI"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=256

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

python3 ablation.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/${TASK}_${MODEL}_ablation.pt" \
    --output_path "output/${TASK}_${MODEL}_ablation.json" \
    --ambig_train_path "calibration_data/${TASK}/train_ambig33_bert.tsv" \
    --easy_train_path "calibration_data/${TASK}/train_easy33_bert.tsv" \
    --dev_path "calibration_data/${TASK}/dev.tsv" \
    --test_path "calibration_data/${TASK}/test.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_evaluate \
    --do_train \
    --mixup \
    --mixup_type easy_ambig 

```
