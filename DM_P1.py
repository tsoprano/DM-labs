from nltk.tokenize import RegexpTokenizer
import os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import collections
import math
import time
start_time = time.time()

corpusroot = './presidential_debates'
stemmer = PorterStemmer()
stopWordList = stopwords.words('english')
txt_dict = {}
all_words = []
tf_dict = {}
N = len(os.listdir(corpusroot)) #no. of debate txt files in the directory

def preprocess(doc):
    doc = doc.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    res = [i for i in tokens if i not in stopWordList]
    stems = [stemmer.stem(token) for token in res]
    return stems

for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close() 
    stems = preprocess(doc)
    all_words.append(stems)
    txt_dict[filename] = stems
    tf_dict[filename] = collections.Counter(stems)

all_words = list(set([item for sublist in all_words for item in sublist]))

def getidf(token):
    df = 0
    for file in txt_dict:
        if token in txt_dict[file]:
            df = df+1
    if df == 0:
        return -1
    return math.log10(N/df) 
added

print("health_idf = %.12f" % getidf("health"))
print("agenda_idf = %.12f" % getidf("agenda"))
print("vector_idf = %.12f" % getidf("vector"))
print("reason_idf = %.12f" % getidf("reason"))
print("hispan_idf = %.12f" % getidf("hispan"))
print("hispanic_idf = %.12f" % getidf("hispanic"))
print("-------------------- \n")


def getweight_notNorm(filename,token):
    weight = 0
    tf = tf_dict[filename][token]
    if tf>0:
        weighted_tf = 1 + math.log10(tf)
        weight = weighted_tf * getidf(token) #raw weight of terms
    return weight

posting_list = {}

#function to populate magnitude dictionary
def get_magnitude(filename):
    print(filename)
    vec = []
    for term in set(txt_dict[filename]):
        w = getweight_notNorm(filename,term) 
        vec.append(w)
        lst = (filename, w)
        if term not in posting_list:
            posting_list[term] = []
        posting_list[term].append(lst)
    vec = np.array(vec)
    mag_vec = np.sqrt(np.sum(vec**2))
    return mag_vec

print("Creating Magnitude dictionary and Posting list--------------------")
magnitude_time = time.time()
magnitude_dict = {}
all_files = os.listdir(corpusroot)
for file in all_files:
    magnitude_dict[file] = get_magnitude(file)

print("Magnitude dictionary for all docs:")
print(magnitude_dict)
print("--- %s seconds to populate Magnitude dictionary and Posting list ---" % (time.time() - magnitude_time))
print("-------------------- \n")

#including in the posting list all the docs that are not included in the term's list
for word,v in posting_list.items():
    doc_list = [tup[0][0] for tup in v]
    not_in_word = [i for i in all_files if i not in doc_list]
    for doc_x in not_in_word:
        posting_list[word].append((doc_x,float(0)))
    posting_list[word].sort(key=lambda tup: tup[1], reverse=True)
    posting_list[word] = posting_list[word][:10] #include just the top 10 highest tf-idf weight for each term in the posting list
    temp_tuple_list = []
    for tup in posting_list[word]:
        temp_tuple_list.append((tup[0], tup[1]/magnitude_dict[tup[0]]))  #normalizing the posting list
    posting_list[word] = temp_tuple_list

def getweight(filename,token):
    weight = 0
    tf = tf_dict[filename][token]
    if tf>0:
        magnitude = magnitude_dict[filename]
        weight = getweight_notNorm(filename,token)/magnitude  #normalized tf-idf weight of terms
    return weight

#function to convert query text to normalized frequency vectors
def queryVec(queryTerm):
    q = []
    stems_q = preprocess(queryTerm)
    count = collections.Counter(stems_q)
    for term in stems_q:
        q.append(1 + math.log10(count[term] if count[term]>0 else 1))
    q = np.array(q)
    q = q/np.sqrt(np.sum(q**2))
    return q

def query(queryTerm):
    queryTermVec = preprocess(queryTerm) #preprocessing the query term which returns the stems
    temp_postList = {} #dictionary which contains the posting list entries of the query term stems
    for term in queryTermVec:
        if term in all_words:
            temp_postList[term] = posting_list[term]
    queryTermVec = [i for i in queryTermVec if i in all_words]  #if query term not in corpus, its ignored
    if len(queryTermVec) == 0:
        return ('None', 0)

    present_in_all = [tup[0] for tup in temp_postList[queryTermVec[0]]] #list of common documents in the top 10 entries for each query term
    for k,v in temp_postList.items():
        curr = [tup[0] for tup in v]
        present_in_all = list(set(curr).intersection(present_in_all))
    # print(present_in_all) 
    
    #All files in temp post list 
    present_in_none = [i for i in all_files if i not in present_in_all] #list of uncommon documents for each query
    
    sim_q_d = 0
    if len(present_in_all)>0:
        sim_dict = {}
        sim_dict_rest = {}
        
        #cosine similarity for all common documents
        for file in present_in_all:
            df = []
            for q in queryTermVec:
                df.append([tup[1] for tup in temp_postList[q] if tup[0]==file])
            df = np.array(df).flatten()
            sim_dict[file] = np.sum(queryVec(queryTerm) * df)
        sim_q_d = sim_dict[max(sim_dict, key=sim_dict.get)]
        (fName, simVal) = (max(sim_dict, key=sim_dict.get), sim_q_d)
        
        #cosine similarity for all the rest documents
        for file in present_in_none:
            df = []
            for q in queryTermVec:
                if len([tup[1] for tup in temp_postList[q] if tup[0]==file])==0:
                    df.append([[tup[1] for tup in temp_postList[q]][-1]])  #if doc not in posting list, take the 10th highest tf-idf weight instead
                else:
                    df.append([tup[1] for tup in temp_postList[q] if tup[0]==file])
            df = np.array(df).flatten()
            sim_dict_rest[file] = np.sum(queryVec(queryTerm) * df)
        sim_list_rest = list(sim_dict_rest.values())
            
        #check if non-common docs have higher cosine similarity    
        for i in sim_list_rest:
            if float(i) > simVal or float(i) == simVal:
                (fName, simVal) = ('fetch more', 0)
    
    return (fName, simVal)  


print("health insurance wall street --> (%s, %.12f)" % query("health insurance wall street"))
print("security conference ambassador --> (%s, %.12f)" % query("security conference ambassador"))
print("particular constitutional amendment --> (%s, %.12f)" % query("particular constitutional amendment"))
print("terror attack --> (%s, %.12f)" % query("terror attack"))
print("vector entropy --> (%s, %.12f)" % query("vector entropy"))
print("--- %s seconds to execute entire program ---" % (time.time() - start_time))
print("-------------------- \n")
