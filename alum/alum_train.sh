#!/bin/bash
# script to train an alum-roberta model.
# by Xiaodong Liu: xiaodl at microsoft.com

PEAK_LR=1e-5
ALUM=alum
TOTAL_UPDATES=100000
WARMUP_UPDATES=10000
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512
MAX_SENTENCES=8
UPDATE_FREQ=2
WORKER=2

fairseq-train
    --fp16 data-bin/wikitext-103
    --task adv_masked_lm
    --criterion adv_masked_lm
    --arch advbert_base
    --sample-break-mode complete
    --tokens-per-sample 512
    --optimizer adam --adam-betas '(0.9,0.98)'
    --adam-eps 1e-6 --clip-norm 0.0
    --lr-scheduler polynomial_decay
    --lr 1e-5
    --warmup-updates 100000
    --total-num-update 100000
    --dropout 0.1
    --attention-dropout 0.1
    --weight-decay 0.01
    --max-sentences 8
    --update-freq 2
    --max-update 100000
    --log-format simple 
    --log-interval 500
    --skip-invalid-size-inputs-valid-test
    --adv_alpha 10

