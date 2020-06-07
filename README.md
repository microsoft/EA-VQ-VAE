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

We first train VQ-VAE with the posterior distribution p(z|x,y).

```shell
cd vq-vae
task=event2mind #event2mind or atomic
train_steps=20000 #20000 for event2mind and 50000 for atomic
mkdir -p log model/$task
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
--model_name_or_path gpt2 \
--data_dir ../data/$task \
--output_dir model/$task \
--do_train \
--z_size 400 \
--max_event_length 64 \
--max_target_length 32 \
--train_batch_size 64 \
--eval_batch_size 128 \
--eval_steps 1000 \
--learning_rate 5e-5 \
--train_steps $train_steps \
--gradient_accumulation_steps 1 2>&1 | tee log/log-$task-train.txt
```

We then calculate true prior distribution of train and dev dataset.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
--model_name_or_path gpt2 \
--load_model_path model/$task/pytorch_model.bin \
--data_dir ../data/$task \
--output_dir model/$task \
--do_label \
--z_size 400 \
--max_event_length 64 \
--max_target_length 32 \
--eval_batch_size 128 2>&1 | tee log/log-$task-test.txt
```



# Train Prior Distribution Estimator

We then train prior distribution estimator p(z|x).

```shell
cd ../estimator
task=event2mind #event2mind or atomic
train_steps=10000 #10000 for event2mind and 20000 for atomic
mkdir -p log model/$task
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
--model_name_or_path roberta-large \
--prior_distribution_dir ../vq-vae/model/$task \
--data_dir ../data/$task \
--output_dir model/$task \
--do_train \
--z_size 400 \
--max_event_length 64 \
--train_batch_size 32 \
--eval_batch_size 64 \
--eval_steps 1000 \
--learning_rate 1e-5 \
--train_steps $train_steps \
--gradient_accumulation_steps 1 2>&1 | tee log/log-$task-train.txt
```

We then calculate approximate posterior distribution of train, dev and test dataset.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
--model_name_or_path roberta-large \
--prior_distribution_dir ../vq-vae/model/$task \
--load_model_path model/$task/pytorch_model.bin \
--data_dir ../data/$task \
--output_dir model/$task \
--do_label \
--z_size 400 \
--max_event_length 64 \
--eval_batch_size 128 2>&1 | tee log/log-$task-test.txt
```



# Train Evidence-Aware Decoder

We finally jointly learn the context distribution p(c|z) and the generator p(y|x,c)

```shell
cd ../generator
task=event2mind #event2mind or atomic
train_steps=20000 #20000 for event2mind and 50000 for atomic
num_evidence=20
mkdir -p log model/$task
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
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
--num_evidence $num_evidence \
--eval_steps 1000 \
--train_batch_size 64 \
--eval_batch_size 128 \
--learning_rate 5e-5 \
--train_steps $train_steps \
--gradient_accumulation_steps 1 2>&1 | tee log/log-$task-train.txt
```

We then obtain topK latent variables for selecting evidences

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
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
--num_evidence $num_evidence \
--eval_batch_size 128  2>&1 | tee log/log-$task-topk.txt
```

Using topK latent variable to select evidences for inference

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
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
--num_evidence $num_evidence \
--eval_batch_size 128  2>&1 | tee log/log-$task-test.txt
```


# Cite
If you find our code useful, please consider citing our paper:
```
@inproceedings{daya2020evidence,
  title={Evidence-Aware Inferential Text Generation with Vector Quantised Variational AutoEncoder},
  author={Daya Guo, Duyu Tang, Nan Duan, Jian Yin, Daxin Jiang and Ming Zhou},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
