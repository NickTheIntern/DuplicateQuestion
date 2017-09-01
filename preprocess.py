import csv
import re
import nltk
import numpy as np
import pickle
sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")

def preprocess(s):
    s =  s.decode(encoding='UTF-8').strip()
    tokenized = []
    for string in sentence_detector.tokenize(s):
        string = re.sub(r"`", " ", string)
    	string = re.sub(r"/", " / ", string)
    	string = re.sub(r"=", " = ", string)
    	string = re.sub(r"~", " ~ ", string)
    	string = re.sub(r"~(\s+~)+", " ~~~ ", string)
    	string = re.sub(r"\.{3,}", "...", string)
	tokenized.extend(nltk.tokenize.word_tokenize(string))
    result = " ".join(tokenized)
    return result.lower()


f1 = open("raw_train.csv",'rb')
f2 = open("raw_test.csv",'rb')

s1 = []
s2 = []
label = []

reader = csv.reader(f1)
for row in reader:
    s1 = s1 + [preprocess(row[3])]
    s2 = s2 + [preprocess(row[4])]
    label = label + [row[5]]

s1 = s1[1:]
s2 = s2[1:]
label = np.array(label[1:])
label = label.astype(int)

one_hot_label = np.zeros((len(s1),2))
one_hot_label[np.arange(len(s1)), label] = 1

pickle.dump((s1,s2,one_hot_label), open("preprocessed_train.pickle",'wb'))


index = 0
s1 = []
s2 = []

reader = csv.reader(f2)
reader.next()
for row in reader:
    s1 = s1 + [preprocess(row[1])]
    s2 = s2 + [preprocess(row[2])]
    index = index + 1
    if index % 200000 == 0:
        pickle.dump((s1,s2), open("preprocessed_test%d.pickle" % (index/200000 - 1),'wb'))
        s1 = []
        s2 = []

pickle.dump((s1,s2), open("preprocessed_test%d.pickle" % (index/200000),'wb'))






