import numpy as np;
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import json
import sys
# from train_model import get_text_langid
import os;
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
# Load the training data
data = sys.argv[2];
saveloc = sys.argv[1];
outfilename = sys.argv[3];
with open(data, 'r') as f:
    data = json.load(f)
def get_text_langid(data):
        X = ['^' + i['text'] + '$' for i in data]
        y = [i['langid'] for i in data]
        return X, y
with open(os.path.join(saveloc, 'vect.pickle'), 'rb') as f:
    vect = pickle.load(f);
with open(os.path.join(saveloc, 'model.pickle'), 'rb') as f:
    model = pickle.load(f);

#now we have our model ready.
X, y = get_text_langid(data);
X_vec = vect.transform(X);
y_pred = model.predict(X_vec);

#writing our outputs:

with open(outfilename, 'w') as f:
    for i in range(len(y_pred)):
        f.write(y_pred[i] + '\n');

#DONE.
print("done outputting")