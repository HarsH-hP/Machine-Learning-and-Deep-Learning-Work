# Restaurant-Review-Analysis
# Using Natural Language Processing
Dataset is tsv file {tab seperated file} because in the reviews there can be comma's so csv file is not good for the model.

All the pre-processing of the data is done which includes removing stopwords and working on stemming method used to get the root words. This is necessary to remove all the common words such as{and, then, etc.}
and converting words as{loved-love, etc.}.

After preprocessing all the unique words are obtained which is used to train the model and predict if the review is positive or negative.
Here the model is trained using classification algorithm as here 1- describes positive review and 0- describes negative review.
The algorithm used is Naive Bayes to train the model.

# Code file includes comments which will help understand the code
