---
layout: post
comments: true
author: Dan Salo
title:  "Explaining Black Box Classifiers"
description: "Yes but why?"
date:   2018-10-24 05:00:00
mathjax: true
---

### Terminology
An **interpretable representation** is one that makes sense to humans. Such formats can be easily reasoned about and compared and discussed. Ultimately, humans can use such representations as input in their own decision making processes.

The mechanisms of **interpretable models** are *transparent* and can be represented interpretably. For example, linear regression is commonly cited as an interpretable model when assuming 1) the input terms are interpretable and 2) the number of inputs terms is sufficiently constrained. If non-linear inputs are added or the number of input terms grows to be unwieldy (e.g. 100's) 

**Explaining** is required when a human desires information in an interpretable representation around the mechanisms of a *non-transparent* or *uninterpretable* or *black-box* model. Explanations can be local and specific to an instance, or they can be global and apply to many instances. The goal of 


### Latent Space



This paper shows that black box classifers have "entangled" representations.

Best paper award at ICML 2019, 
[Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://ai.googleblog.com/2019/04/evaluating-unsupervised-learning-of.html)



[LIME](https://christophm.github.io/interpretable-ml-book/lime.html#lime)

["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf)

[Anchors](https://christophm.github.io/interpretable-ml-book/anchors.html)

[Anchors: High-Precision Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)

[Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/pdf/1811.10154.pdf) by Cynthia Rudin