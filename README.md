# DuplicateQuestion

We modifed the code from https://github.com/tensorflow/models/tree/master/skip_thoughts and trained our model for the task of duplicate question detection.

The code for our model is in code/skip_thoughts/skip_thoughts_model.py. By training another Gradient Boosting classifier on top of our GRU model, we achieved a test accuracy of 86.9%, where the test data is a random split of 5000 examples from the Quora training dataset. See our report for more details.
