# Who Said It On Reddit: Multiclass Classification of Reddit Comments using Neural Networks

Austin Cai, Cynthia Liang, Erick Fidel Siavichay-Velasco

Reddit is a web content aggregation site where users can post and comment on specific community groups called "subreddits" [1]. Understanding the nature of discussions that occur on Reddit can provide us with a better general understanding of the intricacies of social interaction, especially anonymized online social interaction. 

Multiclass classification has been implemented in a variety of fields to assign datapoints to classification buckets. Cascaded classification models (CCMs), or "repeated instantiations of... classifiers...coupled by their input/output variables", have been shown to be effective in improving performance on certain tasks [2]; though there has not been as much success in NLP applications of CCMs [3].

In this project, we propose a classification system for Reddit comments with a neural network model with cascaded input vectors in which several sub-problems are considered simultaneously, similar to that of a CCM. We used this system on a Kaggle dataset, which we filtered down to 50k comments from each of the top 20 commented-on subreddits for May 2015 [4].

We built a neural network classification model using a feature vector with features based on both word embeddings and Naive Bayes probabilities and achieved 41.7% accuracy in classifying comments on the test set, compared to 31.6% accuracy from our baseline model and 60% from our human oracle model.

Possible uses for this work include targeting advertisement posts or comments to subreddits where they may be highly discussed or identifying high-risk subreddits associated with certain types of malicious content, as well as new approaches for future models in Natural Language Processing.

[1] _Reddit_, www.reddit.com/.

[2] Heitz, Geremy, et al. “Cascaded Classification Models: Combining Models for Holistic Scene Understanding.” _Adv. in Neural Inform. Proc. Systems (NIPS)_, 2008, pp. 641–648.

[3] Sutton, Charles, et al. “Joint parsing and semantic role labeling.” _Proceedings of the Ninth Conference on Computational Natural Language Learning_, June 29-30, 2005, Ann Arbor, Michigan.

[4] Reddit. “May 2015 Reddit Comments.” _Kaggle_, 4 June 2019, www.kaggle.com/reddit/reddit-comments-may-2015.
