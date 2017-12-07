import gensim as gs
import numpy as np
import os
import csv
import time
from utils import DataLoader

# read configuration
# assumed number of users with labels
least = 1000

# current feature size
fsize = 100

# number of iteration while training like-embedding model
ite = 50 # if needed

n_vocab = 180000
n_word = 35

d = DataLoader()
d.read_train("data/training_label.txt")
d.read_test("data/testing_data.txt")
d.read_augment("data/training_nolabel.txt")

class iterate_corpus(object):
    def __init__(self, all_sentences):
        self.all_sentences = all_sentences

    def __iter__(self):
        for i in range(len(self.all_sentences)):
            yield gs.models.doc2vec.TaggedDocument(self.all_sentences[i],[i])

# for every data size to build a model
corpus = iterate_corpus(d.augment_sentence + d.test_sentence + d.train_sentence) # iterable corpus used for training 

print("===============")
print("PVDBOW starting...")

start = time.time()

model_pvdbow = gs.models.doc2vec.Doc2Vec(size=fsize,min_count=1,dm=0,window=8,workers=6)
model_pvdbow.build_vocab(corpus)
for _ in range(ite):
    model_pvdbow.train(corpus)
model_pvdbow.save("doc2vec_model_d"+str(fsize)+".doc2vec")

end = time.time()
print("take "+str(end-start)+" seconds")

train_doc2vec = []
for i in range(len(d.train_sentence)):
    this = model_pvdbow.infer_vector(d.train_sentence[i])
    train_doc2vec.append(this)
f = open("PVDBOW_d"+str(fsize)+"_train.csv","w")
w = csv.writer(f)
w.writerows(train_doc2vec)
f.close()

test_doc2vec = []
for i in range(len(d.test_sentence)):
    this = model_pvdbow.infer_vector(d.test_sentence[i])
    test_doc2vec.append(this)
f = open("PVDBOW_d"+str(fsize)+"_test.csv","w")
w = csv.writer(f)
w.writerows(test_doc2vec)
f.close()

print("doc2vec embedding model finished.")