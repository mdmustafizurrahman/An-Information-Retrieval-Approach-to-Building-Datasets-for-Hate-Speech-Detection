## An-Information-Retrieval-Approach-to-Building-Datasets-for-Hate-Speech-Detection

For more details about our hate speech dataset, please read the following research article 

Md Mustafizur Rahman, Dinesh Balakrishnan, Dhiraj Murthy, Mucahid Kutlu, and Matthew Lease, An Information Retrieval Approach to Building Datasets for Hate Speech Detection. [[pdf]](https://openreview.net/forum?id=jI_BbL-qjJN)


## Source codes

1. Pooling --> /codes/pooling.py 
2. Active learning --> /codes/active_learning.py


## Benchmark Models

1. BiLSTM [[1]](#1) [[Source code]](https://github.com/aymeam/User_distribution_experiments/tree/master/Model1_Experiments) 
2. LSTM [[2]](#2) [[Source code]](https://github.com/aymeam/User_distribution_experiments/tree/master/Model2_Experiments)
3. BERT [[3]](#3) [[Source code]](https://github.com/huggingface/transformers)

### The source code for BiLSTM and LSTM used in this project are collected from [[4]](#4) where the authors made necessary correction for those two models. 

## Train and Test sets to Benchmark Models

1. Train.csv --> /data/train_test_sets/
2. Test.csv --> /data/train_test_sets/

## Annotation Interface

Two different annotation interfaces used during pilot and main phases are provided in html format under /interface/ directory. 

## Author Distribution of Tweets

Total Number of Authors: 9534
1. Total number of Author with exactly 1 contribution: 9430
2. Total number of Author with exactly 2 contribution: 97
3. Total number of Author with more than 2 contribution: 7


## References

<a id="1">[1]</a> Sweta Agrawal and Amit Awekar. 2018. Deep learning for detecting cyberbullying across
multiple social media platforms. In European Conference on Information Retrieval. Springer,
141–153.

<a id="2">[2]</a> Pinkesh Badjatiya, Shashank Gupta, Manish Gupta, and Vasudeva Varma. 2017. Deep learning
for hate speech detection in tweets. In Proceedings of the 26th International Conference on
World Wide Web Companion. 759–760.

<a id="3">[3]</a> Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pretraining of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

<a id="4">[4]</a> Aymé Arango, Jorge Pérez, and Barbara Poblete. 2019. Hate speech detection is not as easy as
you may think: A closer look at model validation. In Proceedings of the 42nd international
acm sigir conference on research and development in information retrieval. 45–54.
