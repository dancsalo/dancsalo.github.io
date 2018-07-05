---
layout: post
comments: true
author: Dan Salo
title:  "Deep Bayesian Generative Models for Classification"
description: "How can we leverage generative models with unlabeled data to improve classification accuracy?"
date:   2017-08-08 05:00:00
mathjax: true
---

### Introduction

Here is the question we wish to answer:
"How can we leverage generative models with unlabeled data to improve classification accuracy?"
Before we jump into the math, let's define some terminology.

#### Data and Algorithmic Models
Data Models and Algorithmic Models have key differences as Leo Breiman defines in his seminal paper:
[Statistical Modeling: The Two Cultures](http://www2.math.uu.se/~thulin/mm/breiman.pdf).
I highly recommend digesting it as the truths that Breiman unpacked almost 20 years ago still hold true today.
Any reference to a "model" in the following post will be a type of algorithmic model,
which attempts to predict the response or label $y$ for a given input $x$.

#### Supervised, Unsupervised, and Semi-Supervised Learning
Supervised and unsupervised learning summarize the extremes of the training regimen spectrum for algorithmic models.
Supervised learning equates to training a model with labeled data,
while unsupervised learning expects a model to learn useful patterns without labels.
Semi-supervised learning lies along the continuum in between, where the model employs labeled and unlabeled data
to perform a task.

#### Discriminative and Generative Models
Discriminative and generative models delineate the architectures of algorithmic models.
For a supervised learning task, discriminative models learn the mapping between the set of labels ${Y}_j$ and the set of data ${X}_i$;
a label prediction $\hat{y}$ is made by applying a datum $x$ to the model, which is acting as a complicated function.
On the other hand, generative models first learn a joint distribution of the set of labels ${Y}_j$ and the set of data ${X}_i$;
a label prediction $\hat{y}$ is achieved by factoring that joint distribution into a conditional distribution $p(Y|X=x)$ and then sampling from
that conditional distribution.

#### Bayesians and Frequentists
These differences between generative and discriminative models reflect the differing approaches of the Bayesian and Frequentist schools of thought respectively.
Frequentists believe that the parameters of interest are fixed and unchanging under all realistic circumstances;
therefore, discriminative models ask, "To which category $y$ does this datum $x$ belong?"
Bayesians view the world probabilistically rather than as a set of fixed phenomena that are either known or unknown;
therefore, generative models ask, "Which category distribution $p(y)$ is most likely to generate this datum $x$?
Both statistical schools of thought have developed ways to validate their models, estimate error and variance, and incorporate
pre-existing information, and each has their proponents.
But sometimes it just boils down to [common sense](https://xkcd.com/1132/).


### Mathematics

Let's rephrase the original question into a thesis:
We are going to learn the distribution of data by training a generative model with unlabeled and labeled data,
draw and then learn a mapping from samples drawn to distribution using the labeled data.

The key difference between this paradiam and the normal discriminative model is the injection of
the data distribution in between the data and learned mapping.


section 6 in [An Introduction to Variational Methods
for Graphical Models](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf)


$$\begin{align*}
\tag{1} \log p(X) &= log \int_Z p(X,Z) \\
\tag{2} &= \log \int_Z p(X,Z) \frac{q(Z)}{q(Z)} \\
\tag{3} &= \log \Bigg ( \mathbb{E}_q \bigg [ \frac{p(X,Z)}{q(Z)} \bigg ] \Bigg ) \\
\tag{4} &\geq \mathbb{E}_q \bigg [ \log \frac{p(X,Z)}{q(Z)} \bigg ] \\
\tag{5} &= \mathbb{E}_q \big [ \log p(X,Z) \big ] - \mathbb{E}_q \big [ \log q(Z) \big ] \\
\tag{6} &= \mathbb{E}_q \big [ \log p(X,Z) \big ] - H[Z] \\
\end{align*}$$

1.

2.

### Kulback Lieber Divergence

Jensen's Inequality allows you to upper bound the marginal likelihood.

### Parameter Inference

The key idea to VAE paper is that neural networks can infer these generative statistics.
There is one requisite through which to jump: the reparameterization trick.
