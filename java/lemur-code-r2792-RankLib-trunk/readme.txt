* 
* NOTE!!! this is a fixed version of RankLib that DISABLES caching in coordinate ascent.
*         We have found caching has some subtle bug:
*           1. Sometimes it makes small mistakes.
*           2. However, big blunders do happen as well.
*

Date:		June, 2020.
Version:	2.14
======================================
======================================
1. OVERVIEW

RankLib is a library for comparing different ranking algorithms. In
the current version: 
- Algorithms: MART, RankNet, RankBoost, AdaRank, Coordinate Ascent,
LambdaMART, ListNet and Random Forests. 
- Training data: it allow users to: 
   + Specify train/test data separately
   + Automatically does train/test split from a single input file
   + Do k-fold cross validation (only sequential split at the moment,
   NO RANDOM SPLIT) 
   + Allow users to specify validation set to guide the training
   process. It will pick the model that performs best on the
   validation data instead of the one on the training data. This is
   useful for easily overfitted algorithms like RankNet. 
   + ...
- Evaluation metrics: MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k

====================================================================
2. HOW TO USE

2.1. Binary
Usage: java -jar RankLib.jar <Params>
Params:
  [+] Training (+ tuning and evaluation)
	-train <file>		Training data
	-ranker <type>		Specify which ranking algorithm to use
				0: MART (gradient boosted regression tree)
				1: RankNet
				2: RankBoost
				3: AdaRank
				4: Coordinate Ascent
				6: LambdaMART
				7: ListNet
				8: Random Forests
	[ -feature <file> ]	Feature description file: list features to be considered by the learner, each on a separate line
				If not specified, all features will be used.
	[ -metric2t <metric> ]	Metric to optimize on the training data. Supported: MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k (default=ERR@10)
	[ -metric2T <metric> ]	Metric to evaluate on the test data (default to the same as specified for -metric2t)
	[ -gmax <label> ]	Highest judged relevance label. It affects the calculation of ERR (default=4, i.e. 5-point scale {0,1,2,3,4})
	[ -test <file> ]	Specify if you want to evaluate the trained model on this data (default=unspecified)
	[ -validate <file> ]	Specify if you want to tune your system on the validation data (default=unspecified)
				If specified, the final model will be the one that performs best on the validation data
	[ -tvs <x \in [0..1]> ]	Set train-validation split to be (x)(1.0-x)
	[ -tts <x \in [0..1]> ]	Set train-test split to be (x)(1.0-x). -tts will override -tvs
	[ -kcv <k> ]		Specify if you want to perform k-fold cross validation using ONLY the specified training data (default=NoCV)
	[ -norm <method>]	Normalize feature vectors (default=no-normalization). Method can be:
				sum: normalize each feature by the sum of all its values
				zscore: normalize each feature by its mean/standard deviation
	[ -save <model> ]	Save the learned model to the specified file (default=not-save)
	[ -silent ]		Do not print progress messages (which are printed by default)

    [-] RankNet-specific parameters
	[ -epoch <T> ]		The number of epochs to train (default=100)
	[ -layer <layer> ]	The number of hidden layers (default=1)
	[ -node <node> ]	The number of hidden nodes per layer (default=10)
	[ -lr <rate> ]		Learning rate (default=0.00005)

    [-] RankBoost-specific parameters
	[ -round <T> ]		The number of rounds to train (default=300)
	[ -tc <k> ]		Number of threshold candidates to search. -1 to use all feature values (default=10)

    [-] AdaRank-specific parameters
	[ -round <T> ]		The number of rounds to train (default=500)
	[ -noeq ]		Train without enqueuing too-strong features (default=unspecified)
	[ -tolerance <t> ]	Tolerance between two consecutive rounds of learning (default=0.0020)
	[ -max <times> ]	The maximum number of times can a feature be consecutively selected without changing performance (default=5)

    [-] Coordinate Ascent-specific parameters
	[ -r <k> ]		The number of random restarts (default=2)
	[ -i <iteration> ]	The number of iterations to search in each dimension (default=25)
	[ -tolerance <t> ]	Performance tolerance between two solutions (default=0.0010)
	[ -reg <slack> ]	Regularization parameter (default=no-regularization)

    [-] {MART, LambdaMART}-specific parameters
	[ -tree <t> ]		Number of trees (default=1000)
	[ -leaf <l> ]		Number of leaves for each tree (default=10)
	[ -shrinkage <factor> ]	Shrinkage, or learning rate (default=0.1)
	[ -tc <k> ]		Number of threshold candidates for tree spliting. -1 to use all feature values (default=256)
	[ -mls <n> ]		Min leaf support -- minimum #samples each leaf has to contain (default=1)
	[ -estop <e> ]		Stop early when no improvement is observed on validaton data in e consecutive rounds (default=100)

    [-] ListNet-specific parameters
	[ -epoch <T> ]		The number of epochs to train (default=1500)
	[ -lr <rate> ]		Learning rate (default=0.00001)

    [-] Random Forests-specific parameters
	[ -bag <r> ]		Number of bags (default=300)
	[ -srate <r> ]		Sub-sampling rate (default=1.0)
	[ -frate <r> ]		Feature sampling rate (default=0.3)
	[ -rtype <type> ]	Ranker to bag (default=0, i.e. MART)
	[ -tree <t> ]		Number of trees in each bag (default=1)
	[ -leaf <l> ]		Number of leaves for each tree (default=100)
	[ -shrinkage <factor> ]	Shrinkage, or learning rate (default=0.1)
	[ -tc <k> ]		Number of threshold candidates for tree spliting. -1 to use all feature values (default=256)
	[ -mls <n> ]		Min leaf support -- minimum #samples each leaf has to contain (default=1)

  [+] Testing previously saved models
	-load <model>		The model to load
	-test <file>		Test data to evaluate the model (specify either this or -rank but not both)
	-rank <file>		Rank the samples in the specified file (specify either this or -test but not both)
	[ -metric2T <metric> ]	Metric to evaluate on the test data (default=ERR@10)
	[ -gmax <label> ]	Highest judged relevance label. It affects the calculation of ERR (default=4, i.e. 5-point scale {0,1,2,3,4})
	[ -score <file>]	Store ranker's score for each object being ranked (has to be used with -rank)
	[ -idv ]		Print model performance (in test metric) on individual ranked lists (has to be used with -test)
	[ -norm ]		Normalize feature vectors (similar to -norm for training/tuning)

2.2. Build
An ant xml config. file is included. Make sure you have ant on your machine. Just type "ant" and you are good to go.

==================================================================
3. FILE FORMAT (TRAIN/TEST/VALIDATION)

The file format of the training and test and validation files is the same as for SVM-Rank (http://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html). This is also the format used in the LETOR datasets. Each of the following lines represents one training example and is of the following format:

<line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
<target> .=. <float>
<qid> .=. <positive integer>
<feature> .=. <positive integer>
<value> .=. <float>
<info> .=. <string>

Here's an example: (taken from the SVM-Rank website). Note that everything after "#" are discarded.

3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
2 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B 
1 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C
1 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D  
1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2A  
2 qid:2 1:1 2:0 3:1 4:0.4 5:0 # 2B 
1 qid:2 1:0 2:0 3:1 4:0.1 5:0 # 2C 
1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2D  
2 qid:3 1:0 2:0 3:1 4:0.1 5:1 # 3A 
3 qid:3 1:1 2:1 3:0 4:0.3 5:0 # 3B 
4 qid:3 1:1 2:0 3:0 4:0.4 5:1 # 3C 
1 qid:3 1:0 2:1 3:1 4:0.5 5:0 # 3D
