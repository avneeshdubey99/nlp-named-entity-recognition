# Named Entity Recognition on Tweets (WNUT-16)

Tagging people, companies, places, products and 6 other entity types in noisy tweets, without relying on hashtags. The project compares a **BiLSTM + CRF** model (TensorFlow, word2vec-initialised embeddings) with a fine-tuned **`bert-base-uncased`** model.

## Problem
Twitter carries about 500 million tweets a day. Hashtags are an unreliable signal for what a tweet is about, because they are often missing, misspelled or wrong. Named Entity Recognition (NER) finds the entities a tweet actually mentions, and those mentions can then feed trend detection, brand monitoring and crisis response. Tweets are a hard domain for NER: they are short and informal, full of @mentions, URLs, emoji and slang, and they keep mentioning new entities.

## Dataset
**WNUT-16** (the Twitter NER shared task at W-NUT 2016). It is in CoNLL format with BIO tags over 10 entity types: `person`, `geo-loc`, `company`, `facility`, `product`, `musicartist`, `movie`, `sportsteam`, `tvshow` and `other`.

| Split | Tweets | Tokens | Entities |
|---|---|---|---|
| Train | 2,394 | 46,469 | 1,496 |
| Test | 3,850 | 61,908 | 3,473 |

- About 95% of tokens are `O`, so the models are evaluated with **entity-level F1** (`seqeval`) rather than token accuracy.
- 52% of test entity tokens never appear in train, so handling unseen words is the central challenge.

## Approach
**1. BiLSTM + CRF (TensorFlow / Keras)**
- Tweet-aware normalisation (URLs, @mentions, hashtags, digits) and a Keras `Tokenizer`.
- Embeddings initialised from a skip-gram **word2vec** model, plus a casing-feature embedding.
- A (Bi)LSTM that feeds a **linear-chain CRF**, written in plain TensorFlow (forward-algorithm loss, Viterbi decoding).
- Word dropout for robustness to unknown words, and early stopping on validation F1.

**2. BERT (`bert-base-uncased`, Hugging Face)**
- WordPiece tokenisation with label alignment: the first sub-token of each word gets the word's label, and the rest are ignored.
- Hyperparameter runs across optimisers (AdamW, Adafactor), learning rates, epoch counts and early stopping.
- Sub-token predictions are merged back into words. The model is saved with `save_pretrained` and tested on new sentences.

## Results (test set, entity-level F1)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| BiLSTM + CRF (best config) | 0.393 | 0.180 | 0.247 |
| BERT (best run) | _TBD_ | _TBD_ | _TBD_ |

In the BiLSTM experiments, casing features and word dropout raised test F1 from 0.16 to 0.24. Bidirectionality and the CRF made much less difference at this data size.

## How to run
Open `NER_Twitter_WNUT16.ipynb` in Google Colab with a T4 GPU, upload the two `.conll` files, and select **Runtime → Run all**. A full run takes about 30–40 minutes.

> The assignment specifies TensorFlow 2.15, which needs Python 3.11 or older. On newer Python versions the notebook uses the installed TensorFlow with `tf_keras` (the Keras 2 API), and the same code runs unchanged.

## Tech stack
Python · TensorFlow/Keras · PyTorch · Hugging Face Transformers · gensim · seqeval · scikit-learn · pandas · matplotlib

## Acknowledgements
Dataset: Strauss et al. (2016), *Results of the WNUT16 Named Entity Recognition Shared Task*, W-NUT Workshop.

**Author:** Avneesh Dubey, MS Data Science, Texas A&M University
