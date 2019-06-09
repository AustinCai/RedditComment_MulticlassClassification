Hi Jennie,

Here is a quick tour of our repository. 

"neural_net_model" contains code for our final classification model. The "logistic_regression_model"
contains code for an intermediate model we implemented while experimenting, and "naive_bayes_baseline"
contains our baseline model. 

Here are the important files in each directory. 

=== neural_net_model ===
constants.py -- a helper file
helper.py -- a helper file 
nn_multiclass.py -- our original neural net implementation, without tuned hyperparameters
nn_multiclass_ipython.ipynb -- our final neural net implementation
word_embeddings.py -- helper function to return a dictionary of word embeddings sourced from GloVe
Xy_dump -- creates feature and label vectors out of our input dataset 

=== naive_bayes_baseline ===
nb.py -- contains the naive bayes implementation
nb_helper.py -- a helper file, contains the bulk of the actual calculations
nb_constants -- a helper file

=== linear_regression_model ===
line_reg.py -- contains the implementation

Our code implementation relies on a lot of pkl dumps to store intermediate calculations. However, these
data files are too large to be uploaded to github and were stored in directories outside of our repo. Thus
many of our files will not run unless you replicate our directory structure and run our files in the right
sequence. If you are interested in doing so, please let us know and we can help you get the environment
set up. 

Best,
Austin, Cynthia, and Erick



