Execute: python HW2.py [train_directory] [test_directory] -[Data Model Type] -[Algorithm]

example: python HW2.py "all_data/hw2_train1/train/" "all_data/hw2_test1/test/" -B -D

(This will create Bernoulli datasets and test the Discrete Naive Bayes algorithm using "all_data/hw2_train1/train/" and "all_data/hw2_test1/test/" folders containing training and testing examples respectively. The directories should have separate folders labelled as "ham" and "spam" containing the .txt files as individual examples, just like the directory structure provided in the assignment.)

Data Model Types-
	-B : Bernoulli Data Model
	-W : Bag-of-words Data Model (default)

Algorithms-
	-D : Discrete Naive Bayes 	(works on Bernoulli Data Model)
	-M : Multinomial Naive Bayes 	(works on Bag-of-words Data Model)
	-L : Logistic Regression	(works on both data models)
	-S : SGD Classifier (default)	(works on both data models)
