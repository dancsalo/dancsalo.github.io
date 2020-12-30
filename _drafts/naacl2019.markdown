---
layout: post
comments: true
author: Dan Salo
title:  "Adversarial Transfer Training in NLP"
description: "Top papers and overall themes"
date:   2019-07-02 05:00:00
mathjax: true
---

I had the chance to attend [NAACL 2019](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng_computationallinguistics), and the talks I attended and the conversations I had inspired me to put together a project that combines transfer learning, social media, and cyber security.

#### Transfer Learning Tutorial
[Sebastian Ruder](http://ruder.io/) and his co-authors from AI2, CMU, and Huggingface marched through 220 [slides](https://docs.google.com/presentation/d/1YYiSlSqRJzNHpalevPcMY918Npc9wzcKnZbSNcb6ptU/edit?usp=sharing) with practical tips and tricks for applying this new class of Transformer-based language models, notably [BERT](https://www.aclweb.org/anthology/N19-1423), to
particular target tasks. I give my brief summary of it below.

The goal of transfer learning is to improve the performance on a target tasksby applying knowledge gained through sequential or simultaneous training on a set of source tasks, as summarized by the diagram below from
[A Survey of Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf):

<div class="imgcap">
<img src="/assets/naacl19/transfer.png">
<div class="thecap">Traditional Machine Learning vs. Transfer Learning</div>
</div>

There are three general keys to successful transfer learning:
##### 1) Finding the set of source tasks that produce generalizble knowledge
Learn higher order concepts are generalizble.
In image processing, those concepts are lines, shapes, patterns.
In natural language processing, those concepts are syntax, semantics, morphology, subject verb agreement.

These tasks also need data
Neural Inference
Machine Translation


##### 2) Selecting a method of knowledge transfer
##### 3) Combining the generalizable and specific knowledge 

First point: finding the right set of source tasks. It's important!
Language modeling has been the task of choice for a while now.

Second point: the transfer medium has been maturing over the years.
Word2vec and skip thoughts stored knowledge in a produced vector
Language models _are_ the generalized knowledge. Quite the paradigm shift!

Third point: contextual neural models on language modeling tasks then require the slow introduction of target-specific language.

Optimization.
    Freezing all but the top layer. Long et al [Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/pdf/1502.02791.pdf)
    Chain-thaw, training on layer at a time [Using millions of emoji occurrences to learn any-domain representationsfor detecting sentiment, emotion and sarcasm](https://www.aclweb.org/anthology/D17-1169)
    Gradually unfreezing Howard et al [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
    Sequential unfreezing with hyperparameter tuning Felbo et al [https://arxiv.org/pdf/1902.10547.pdf](An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models)


#### Probing Language Models

PREPRINT
[Visualizing and Measuring the Geometry of BERT](https://arxiv.org/abs/1906.02715)
[Blog post](https://pair-code.github.io/interpretability/bert-tree/)

POSTER ML & SYNATX
[Understanding Learning Dynamics Of Language Models with SVCCA](https://arxiv.org/pdf/1811.00225.pdf)

MACHINE LEARNING 2019 NAACL
[A structural Probe for Finding Syntax in Word Representations](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf)

Applications 2019 NAACL
[Attention is not Explanation](https://arxiv.org/abs/1902.10186)
[The emergence of number and syntax units in LSTM language models](https://arxiv.org/abs/1903.07435)
[Neural Language Models as Psycholinguistic Subjects: Representations of Syntactic State](https://arxiv.org/pdf/1903.03260.pdf)



#### Cybersecurity
[Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems](https://arxiv.org/pdf/1903.11508.pdf)


## Update:
[Vector of Locally-Aggregated Word Embeddings (VLAWE):A Novel Document-level Representation](https://arxiv.org/pdf/1902.08850.pdf)