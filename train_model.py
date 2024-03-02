import numpy as np;
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import json
import sys
import time
import os
# Load the training data
train_start = time.time();
#First argument will be the data location
data_dir = sys.argv[1];
with open(os.path.join(sys.argv[1],'train.json'), 'r', encoding='utf-8') as fp:
        train_data = json.load(fp)
with open(os.path.join(sys.argv[1],'valid_new.json'), 'r', encoding='utf-8') as fp:
        valid_new_data = json.load(fp)
with open(os.path.join(sys.argv[1],'valid.json'), 'r', encoding='utf-8') as fp:
        valid_data = json.load(fp)
saveloc = sys.argv[2];
#Get the X and Y components
def get_text_langid(data):
        X = ['^' + i['text'] + '$' for i in data]
        y = [i['langid'] for i in data]
        return X, y
print("opening data");
X_train, y_train = get_text_langid(train_data);
X_valid, y_valid = get_text_langid(valid_data);
X_valid_new, y_valid_new = get_text_langid(valid_new_data);

#Now we combine them all into one dataset first.
X_train = X_train + X_valid + X_valid_new + X_valid_new;
y_train = y_train + y_valid + y_valid_new + y_valid_new;
del train_data, valid_data, valid_new_data; #why would we need thi snow.
print("data opened");
#using 2 of the new valid datasets to train the model here, since we know they are mostly correct.
## Tokenizers
def pentagram(text):
    s = text.lower();
    return [s[i:i+5] for i in range(len(s)-4)];
def split_by_space(text): #A global function for consistent tokenization.
    return text[1:-1].lower().split(); #simple for now.
def word_and_pentagram(text):
    s = text.lower();
    return split_by_space(s) + pentagram(s);
def hexagram(text):
    s = text.lower();
    return [s[i:i+6] for i in range(len(s) - 5)];
def hexa_penta(text):
    return hexagram(text) + pentagram(text);

## Creating the vectorizer
print("vectorizing...");
start = time.time();
## CHANGE THIS TO HEXA_PENTA, NOT SPLIT_BY_SPACE.
vect = CountVectorizer(tokenizer=hexa_penta, token_pattern=None);
vect.fit(X_train);
X_train_vec = vect.transform(X_train);
print("done vectorizing, time taken: ", time.time() - start, " seconds");
with open(os.path.join(saveloc,"vect.pickle"), "wb") as f:
    pickle.dump(vect, f, protocol=pickle.HIGHEST_PROTOCOL);
print("saved vectorizer");
def run_tests(model):
    y_pred = model.predict(X_train_vec);
    return y_pred;
laplace_smoothing = 0.015;
def train_once(y_to_train,nb = None):
    partial = True;
    if(nb == None):
        nb = MultinomialNB(alpha=laplace_smoothing)
        partial = False;
    if(partial):
        nb.partial_fit(X_train_vec, y_to_train);
    else:
        nb.fit(X_train_vec, y_to_train);
    return nb;
y_soft = [y_train];
print("training...");
start = time.time();
nb = train_once(y_to_train=y_train); #initially we train on the original dataset

#Now to account for the noise in the original dataset, I will train the model additionally on the predictions as well in an interative process.
tot_iter = 80;
for i in range(tot_iter):
    #if(i%10 == 0):
    iter_start = time.time();
    print(tot_iter - i, " iterations left");
    y_new = run_tests(nb);
    # y_soft.append(y_new); #Why do we even need to append tho
    nb.feature_count_ = 1.1 * nb.feature_count_; #This is to give more weightage to the initial training data.
    nb = train_once(y_to_train=y_new, nb=nb);
    print("time for this iteration: ", time.time() - iter_start, " seconds");
print("done training, time taken: ", time.time() - start, " seconds");
print("Saving...");
with open(os.path.join(saveloc,"model.pickle"), "wb") as f:
    pickle.dump(nb, f, protocol=pickle.HIGHEST_PROTOCOL);
print("Model trained and saved, total time taken: ", time.time() - train_start, " seconds");

