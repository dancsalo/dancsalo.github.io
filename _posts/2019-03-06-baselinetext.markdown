---
layout: post
comments: true
author: Dan Salo
title:  "Baselines for Text Classification"
description: "With so many new advances in NLP last year, how high is the lower bound now?"
date:   2019-03-06 11:00:00
mathjax: true
---
## Introduction
If you have kept up with the barage of NLP advances in 2018, kudos to you. I started digging into the literature after 
Sebastian Ruder made his [bold declaration](http://ruder.io/nlp-imagenet/),
and I feel like I've been trying to catch up ever since.
[Jay Alammar blog's](http://jalammar.github.io/illustrated-bert/) is one
of the best resources I have found towards understanding
[BERT](https://arxiv.org/pdf/1810.04805.pdf),
[ELMo](https://arxiv.org/pdf/1802.05365.pdf),
[ULMFit](https://arxiv.org/pdf/1801.06146.pdf),
and the [Transformer module](https://arxiv.org/pdf/1706.03762.pdf).
For the purposes of this post, just know
 that this new class of language models generate word vectors that take into account the
_sentence context_, unlike [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf)
 or [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
 and more elegantly than [CoVe](https://arxiv.org/pdf/1708.00107.pdf). 
 As a side note, **[context2vec](http://www.aclweb.org/anthology/K16-1006)** deserves special recognition 
 since it first showed the power of using bidirectional language models to produce contextual word vectors
 yet received little recognition from the community.

Two questions I'll tackle in this post:
1. **How significantly does text classification accuracy improve when we swap out non-contextual word vectors for contextual
 word vectors in baseline architectures?**
2. **How does the text classification accuracy of a baseline architecture with
 BERT word vectors compare to a fine-tuned BERT model?**   

### Baseline Architectures

So what is a baseline architecture for text classification? Most text classification problems
involve labeling multi-word phrases.
A general-purpose baseline architecture transforms
the phrases or sentences into fixed-length representations and
then learns a simple classifier (usually an MLP) on top.
The following papers showcase several flavors of these transformations:


[A Simple But Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx) (2017) proposed
a weighted average of the set of word vectors in a sentence, $S$, based roughly on their inverse frequency in the training data, $p(w)$.
(The authors also only use the first singular vector of each sentence vector after applying PCA to the 
matrix of training sentences). 

$$\mathbf{v_s} \leftarrow \frac{1}{|S|} \sum_{w \in S} \frac{a}{a + p(w)} \mathbf{v_w}$$

[Concatenated Power Mean Word Embeddings
as Universal Cross-Lingual Sentence Representations](https://arxiv.org/pdf/1803.01400.pdf) (2018) generalized the idea
of averaging word vectors by replacing the arithmetic mean with the power mean. It also introduced the idea of 
concatenating various vectors for a more expressive representation. 

$$\mathbf{v_s} \leftarrow  \left( \frac{\mathbf{v_1}^p + \cdots + \mathbf{v_w}^p}{|s|} \right)^{\frac{1}{p}} \hspace{1cm} p \in \mathbb{R} \cup \{ \pm \infty \}, w = |S|$$


[No Training Required: Exploring Random Encoders for Sentence Classification](https://arxiv.org/pdf/1901.10444.pdf) (2019)
showed that transforming word vectors into _random_ high-dimensional vector spaces and then
pooling them into sentence vectors produces a strong baseline for text classification problems.

$$\mathbf{v_s} \leftarrow \sum_{w \in S} W \cdot \mathbf{v_w} \hspace{1cm} W \in \mathbb{R} $$

In the post, I will be applying a simple arithmetic average to word vectors to produce sentence embeddings. 

### Classy Flair

[Flair](https://github.com/zalandoresearch/flair) is a new NLP framework built on PyTorch for text classification.
It is maintained by (one research scientist at) Zalando Research. I choose to build off of this particular framework
because, at the time, it had the simplest interface for generating contextual word vectors from all of the new models.
The code for all of the experiments in this post and the building blocks for any custom
baseline classifier that you might want to create can be found in the [classyflair](https://github.com/dancsalo/classyflair)
repo. 

As a side note, if I were to run these experiments from scratch again, 
I would opt for the [fastai framework](https://github.com/fastai/fastai), as it is also built on PyTorch but now supports
more models, boasts more contributors, and closes issues faster than Flair.

### PyTorch vs. Tensorflow
Both Flair and fastai run on PyTorch. It took me a while to come around to PyTorch, but now I am a big fan.

When I first got into deep learning in 2016, Theano and Caffe2 were the most widely-used frameworks.
I was working on an object detection project, and the [Caffe2 Zoo](https://github.com/caffe2/models) had the most state-of-the-art pre-trained models at the time.
But because of my limited knowledge of C/C++ and my familiarity with Python, I looked to other frameworks.
PyTorch was not even subversioned yet. Theano, which all of the older
graduate students in the lab were using, already seemed long in the tooth and unwieldy.
So, I opted to use the newly-announced TensorFlow
and hoped that a community would rally around it with lots of cool models and updates.

Indeed, with the backing of Google engineering and [one particularly active developer on StackOverflow](https://stackoverflow.com/users/3574081/mrry),
the community blossomed. I immediately fell in love with TensorBoard,
which simplified the process of selecting a model, an optimizer, and a loss function,
but the difficulty in inspecting tensors during training and the awkwardness of defining a separate
test network were constantly frustrating. 

These frustrations stemmed from that fact that TensorFlow creates a static computational graph under the hood.
 PyTorch, on the other hand, creates a dynamic computational graph
that is defined at runtime, which facilitates variable-length inputs and simple inspection and supports
native Python debugging. Static graphs are natively faster in production, 
but Facebook has responded by providing [ways to productionize](https://github.com/facebookresearch/pytext) PyTorch code
that optimize it for speed.
And while PyTorch still lags behind in the native visualization category, the community has responded with
[alternatives](https://github.com/lanpa/tensorboardX).

So I've switched to PyTorch for the time being. As I see it, the TensorFlow syntax tries to allow data scientists to
tinker _and_ engineers to optimize at the cost of readability and modularity.
PyTorch has separated those concerns, and a [significant increase 
in adoption](https://www.reddit.com/r/MachineLearning/comments/9kys38/r_frameworks_mentioned_iclr_20182019_tensorflow/)
has resulted.

### Fine-tuning BERT
[BERT](https://arxiv.org/pdf/1810.04805.pdf) has received the most attention out of all the models in this new class. 
The authors report state-of-the-art results on a number of NLP datasets by
fine-tuning the unsupervised language model _without any additional architectural modifications_. On one hand, such results
implicitly show the expressive power of stacked Transformers; on the other hand, IRL maintaining a separate fine-tuned
model for each task would be cumbersome. We would like to ensure the the juice from fine-tuning is worth the squeeze,
which motivates the second question of this post.

I won't be fine-tuning a BERT model in this post but rather referring to the published results.

## GLoVe vs. BERT in Baseline

Now for our first question: _How significantly does text classification accuracy improve when we swap out non-contextual
word vectors for contextual word vectors in baseline architectures?_

### Dataset

We will use the [UCI Sentence Classification](http://archive.ics.uci.edu/ml/datasets/Sentence+Classification) corpus
in this section. The data set contains sentences from the abstract and introduction of 30 articles from
the biology, machine learning and psychology domains. Each sentence is labeled by 3 reviewers as 
describing the specific goal of the paper (AIM), the authors' own work (OWN), contrasting previous work (CONT),
past work that provides the basis for the currnent work (BASE), or miscellaneous (MISC).

For simplicity, only the labels from the first reviewer are used for each article in `labeled_articles/`.

### Experiments

To preprocess the UCI dataset, first run:
```python
python main.py -m preprocess -p SciArticles
```
To train a one-layer MLP on top of GLoVe vectors, run:
```python
python main.py -m train -p SciArticles -w glove -a OneLayerMLP -r 0.007 -e 200
```
To train a two-layer MLP on top of GLoVe vectors, run:
```python
python main.py -m train -p SciArticles -w glove -a TwoLayerMLP -r 0.008 -e 200
```
To train a two-layer MLP with dropout on top of GLoVe vectors, run:
```python
python main.py -m train -p SciArticles -w glove -a TwoLayerMLPdp -r 0.01 -e 200
```
Finally, to train a two-layer MLP with dropout on top of BERT vectors, run:
```python
python main.py -m train -p SciArticles -b bert-base-uncased -a TwoLayerMLPdp -r 0.02 -e 200
```
Below is a table of the results;
[micro and macro F1 scores](https://sebastianraschka.com/faq/docs/multiclass-metric.html) are the multi-class analogs
to the [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) for binary
classification.

| Model | Micro F1 Scores | Macro F1 Scores |
|-------|--------|--------|
| *GLoVe*, 1 layer | 0.6268  | 0.1254|
| *GLoVe*, 2 layer | 0.6268  | 0.1254|
| *GLoVe*, 2 layer, dp | 0.6268 | 0.1254|
| *BERT*, 2 layer, dp | 0.7368| 0.4770 |

## Baseline BERT vs. Fine-tuned BERT

Now for our second question: _How does the text classification accuracy of a baseline architecture with
 BERT word vectors compare to a fine-tuned BERT model?_

### Dataset

The [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) is an extension of the
[Movie Review](https://www.cs.cornell.edu/people/pabo/movie-review-data/) data set but with train/dev/test splits
provided along with granular labels (SST-1) and binary labels (SST-2). The authors of 
[BERT](https://arxiv.org/pdf/1810.04805.pdf) report an accuracy of $93.5\%$ on the SST-2 test set.

### Experiments
Let's see how close to $93.5\%$ we can get with a baseline model with BERT word vectors.
To preprocess the SST-2 dataset, first run:

```python
python main.py -m preprocess -p Sst2
```
Then to train a two-layer MLP with dropout with GLoVe and BERT vectors concatenated as inputs,
run several iterations of the following:
```python
python main.py -m train -p Sst2 -w glove -b bert-base-uncased -a TwoLayerMLPdp -r 0.03 -e 200
```

We get an accuracy of $87.1\% \pm 0.06$. I also tried learning character embeddings (by adding the `-c` flag), but the 
improvement was negligible.

## Conclusions

**How significantly does text classification accuracy improve when we swap out non-contextual word vectors for contextual
 word vectors in baseline architectures?** By over 10 points on the Micro F1 Score and over 30 points on the Macro F1 Score
 for the UCI Sentence Classification corpus. Every classification task will be different, but swapping out non-contextual
 word vectors for contextual ones is a no-brainer if you can afford the computation.
 The stagnation of the different MLP models trained
 on top of GLoVe vectors shows that the input vectors' expressiveness was the rate-limiting factor in performance.
 
 **How does the text classification accuracy of a baseline architecture with
 BERT word vectors compare to a fine-tuned BERT model?** The baseline model got within 6 points of the reported 
 accuracy of the fine-tuned model, which is decent, but shows that fine-tuning can add considerable improvements.
 [Evaluation of sentence embeddings in downstream and linguistic probing tasks](https://arxiv.org/pdf/1806.06259.pdf)
 evaluated many different input vectors other than BERT with MLP baseline models, but the highest accuracy they report
 on SST-2 is $86.71\%$ with all 5 ouput vectors from the ELMo model concatenated together. The BERT baseline 
 model reported here (using only the top layer) edges out the ELMo model by a few tenths of a percentage point. 
 It would be interesting to experiment with the different layers of BERT and compare any performance improvements.
 
BERT certainly made a big splash last year, much in the same way AlexNet and VGG did years ago; if the field of NLP
 continues down the path blazed by the image analysis community, I expect to see simplifications and generalizations of
 BERT and its counterparts (à la GoogLeNet and ResNet). [Pay Less Attention with Lightweight and Dynamic Convolutions](https://arxiv.org/pdf/1901.10430.pdf),
 which was accepted as an [Oral at ICLR](https://www.reddit.com/r/MachineLearning/comments/a8nqn3/riclr_oral_pay_less_attention_with_lightweight/)
 this year, is already making strides towards simplifying the Transformer architecture.
 I'm excited to see what other advances come around in 2019!
 