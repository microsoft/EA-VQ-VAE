# Introduction

This repo provides the code for the ACL 2020 paper "Evidence-Aware Inferential Text Generation with Vector Quantised Variational AutoEncoder"



# Requirements

- pip install torch==1.4.0

- pip install gdown transformers==2.8.0 filelock nltk

  

# Download Dataset

###  1.Download Evidence

```shell
cd data
gdown https://drive.google.com/uc?id=1l8o0Itcr-MqKAdMxELSWGd1TnEF8eyXu
cd ..
```

Or you can download searched evidence from [website](https://drive.google.com/open?id=1l8o0Itcr-MqKAdMxELSWGd1TnEF8eyXu) to data folder.

### 2.Download and Preprocess ATOMIC Datasets

```shell
cd data
bash get_atomic_data.sh
python preprocess-atomic.py
cd ..
```

### 3.Download and Preprocess Event2Mind Datasets

```shell
cd data
bash get_event2mind_data.sh
python preprocess-event2mind.py
cd ..
```



# Train Vector Quantised-Variational AutoEncoder (VQ-VAE)

We first train VQ-VAE with the posterior distribution $q_\phi(z|x,y)$.

```shell
cd vq-vae
task=event2mind #event2mind or atomic
train_steps=10000 #10000 for event2mind and 25000 for atomic
mkdir -p log model/$task
python run.py \
--model_name_or_path gpt2 \
--do_train \
--data_dir ../data/$task \
--z_size 400 \
--output_dir model/$task \
--max_event_length 64 \
--max_target_length 32 \
--eval_steps 1000 \
--train_batch_size 32 \
--warmup_steps 0 \
--eval_batch_size 64 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps $train_steps \
--gradient_accumulation_steps 2 2>&1 | tee log/log-$task-train.txt
```

We then calculate true prior distribution of train and dev dataset.

```shell
python run.py \
--model_name_or_path gpt2 \
--load_model_path model/$task/pytorch_model.bin \
--do_label \
--data_dir ../data/$task \
--z_size 400 \
--output_dir model/$task \
--max_event_length 64 \
--max_target_length 32 \
--eval_steps 1000 \
--train_batch_size 32 \
--warmup_steps 0 \
--eval_batch_size 64 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps $train_steps \
--gradient_accumulation_steps 2 2>&1 | tee log/log-$task-test.txt
```



# Train Prior Distribution Estimator

We then train prior distribution estimator $p_\theta(z|x)\sim q_\phi(z|x,y)$.

```shell
cd ../estimator
task=event2mind #event2mind or atomic
train_steps=20000 #20000 for event2mind and 40000 for atomic
mkdir -p log model/$task
python run.py \
--model_name_or_path roberta-large \
--prior_distribution_dir ../vq-vae/model/$task \
--do_train \
--data_dir ../data/$task \
--z_size 400 \
--output_dir model/$task \
--max_event_length 64 \
--eval_steps 2000 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 1e-5 \
--train_steps $train_steps \
--gradient_accumulation_steps 1 2>&1 | tee log/log-$task-train.txt
```

We then calculate approximate posterior distribution of train, dev and test dataset.

```shell
python run.py \
--model_name_or_path roberta-large \
--prior_distribution_dir ../vq-vae/model/$task \
--load_model_path model/$task/pytorch_model.bin \
--do_label \
--data_dir ../data/$task \
--z_size 400 \
--output_dir model/$task \
--max_event_length 64 \
--eval_steps 2000 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 1e-5 \
--train_steps $train_steps \
--gradient_accumulation_steps 1 2>&1 | tee log/log-$task-test.txt
```



# Train Evidence-Aware Decoder

We finally jointly learn the context distribution $p_s(c|z)$ and the generator $p_m(y|x,c)$

```shell
cd ../generator
task=event2mind #event2mind or atomic
train_steps=20000 #20000 for event2mind and 50000 for atomic
mkdir -p log model/$task
python run.py \
--model_name_or_path gpt2 \
--data_dir ../data/$task \
--codebook_path ../vq-vae/model/$task/codebook.bin \
--posterior_dir ../vq-vae/model/$task \
--prior_dir ../estimator/model/$task \
--do_train \
--z_size 400 \
--output_dir model/$task \
--max_evidence_length 64 \
--max_event_length 64 \
--max_target_length 32 \
--num_evidence 25 \
--eval_steps 2000 \
--train_batch_size 16 \
--warmup_steps 0 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps $train_steps \
--gradient_accumulation_steps 2 2>&1 | tee log/log-$task-train.txt
```

We then obtain topK latent variables for selecting evidences

```shell
python run.py \
--model_name_or_path gpt2 \
--load_model_path model/$task/pytorch_model.bin \
--data_dir ../data/$task \
--codebook_path ../vq-vae/model/$task/codebook.bin \
--posterior_dir ../vq-vae/model/$task \
--prior_dir ../estimator/model/$task \
--do_topk \
--z_size 400 \
--output_dir model/$task \
--max_evidence_length 64 \
--max_event_length 64 \
--max_target_length 32 \
--num_evidence 25 \
--eval_steps 2000 \
--train_batch_size 16 \
--warmup_steps 0 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps $train_steps \
--gradient_accumulation_steps 2 2>&1 | tee log/log-$task-topk.txt
```

Using topK latent variable to select evidences for inference

```shell
cd ../generator
task=event2mind #event2mind or atomic
train_steps=20000 #20000 for event2mind and 50000 for atomic
mkdir -p log model/$task
python run.py \
--model_name_or_path gpt2 \
--load_model_path model/$task/pytorch_model.bin \
--data_dir ../data/$task \
--codebook_path ../vq-vae/model/$task/codebook.bin \
--posterior_dir ../vq-vae/model/$task \
--prior_dir ../estimator/model/$task \
--do_test \
--z_size 400 \
--output_dir model/$task \
--max_evidence_length 64 \
--max_event_length 64 \
--max_target_length 32 \
--num_evidence 25 \
--eval_steps 2000 \
--train_batch_size 16 \
--warmup_steps 0 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps $train_steps \
--gradient_accumulation_steps 2 2>&1 | tee log/log-$task-topk.txt
```



