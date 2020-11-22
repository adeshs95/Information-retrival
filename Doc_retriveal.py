import numpy as np 
import os 
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer 
from nltk.stem import WordNetLemmatizer 
import string
from bs4 import BeautifulSoup
import heapq
from operator import itemgetter
import math
import pickle
from scipy.spatial import distance
import sys

def preprocessing(document): #To remove punctuations and stop words and perform casefolding and lemmetization
         
    tokenizer = TweetTokenizer() 
    token_list = tokenizer.tokenize(document) 
    
    
    table = str.maketrans('', '', '\t') # Remove punctuations. 
    token_list = [word.translate(table) for word in token_list] 
    punctuations = (string.punctuation).replace("'", "") 
    trans_table = str.maketrans('', '', punctuations) 
    stripped_words = [word.translate(trans_table) for word in token_list] 
    token_list = [str for str in stripped_words if str] 
    
  
    
    token_list =[word.lower() for word in token_list] # Case folding
    stop_words = set(stopwords.words('english')) #stopwords
    
    filtered_list = [] 
  
    for w in token_list: 
        if w not in stop_words: 
            filtered_list.append(w) 
    

    
    #filtered_list = [t for t in token_list if t not in stop_words] #checking and removal of stop word
    
    filtered_Document =[]
    lemmatizer = WordNetLemmatizer()

    
    for term in filtered_list: #for every term 

        filtered_Document .append(lemmatizer.lemmatize(term) )      #perfroming lemmetization
        
   
    return filtered_Document #free from unwanted information

def TF(All_terms , Corpus):#To get the document frequency and posting list

    
    index = cnt =0
    term_doc_count = []
    doc_tf={}
    
    for term in All_terms:

        term_doc_count.append(0)
        doc_tf [term]=[]
        doc_index=0

        for Doc in Corpus.values():

            cnt= Doc.count(term)
            
            if cnt > 0:
                pair = doc_index , cnt
                doc_tf[term].append(pair)
                term_doc_count[index] += 1

            doc_index+=1
            
        index +=1

    return term_doc_count,doc_tf

def idf(All_terms , N,term_doc_count,doc_tf):#to get idf for terms

    IDF = {}
    index = 0
    
    for Doc_count in term_doc_count:  #math.log(total_docs/doc_count)

        IDF[All_terms[index]] = math.log(N/Doc_count)
        #weight.append(IDF[index] * math.log(1+Doc_count))
        index +=1



    inv_pos_index={}
    for i in range(len(All_terms)):

        key = All_terms[i],IDF[All_terms[i]]
        inv_pos_index[key] = doc_tf[All_terms[i]]

    return inv_pos_index ,IDF

def get_champ_list_local(doc_tf):

    champ_list = {}
    for term in doc_tf.keys():
        champ_list[term] = []
        
        champ_list[term] = ([x[0] for x in heapq.nlargest(50 ,doc_tf[term],key=itemgetter(1))])
        
    return champ_list

def get_champ_list_global(inv_pos_index, IDF ,gd):

    score_gd ={}
    
    index=0
    champ_list_global={}
    for key in inv_pos_index.keys():

        score_gd[key[0]]={}
        

        for Doc_tf in inv_pos_index[key]:
            score_gd [key[0]][Doc_tf[0]] = gd[Doc_tf[0]] + IDF[key[0]]*math.log(1+Doc_tf[1]) #dict to store scores for each term doc pair
            
            index+=1 

    for term in score_gd.keys():
        champ_list_global[term] = []
        champ_list_global[term] = ([x[0] for x in heapq.nlargest(50,score_gd[term].items(),key=itemgetter(1))])
        
    return champ_list_global ,score_gd

def T1(query , IDF ,  Doc_tf):#getting top 10 docs from for the query from the inverted index

    score_doc ={}
    Length ={}
    len_que =0;
    for t in query:

        pos_list = Doc_tf[t]
        len_que += (IDF[t])**2
        for did,tf in pos_list:

            if did not in score_doc.keys():
                score_doc[did] = 0
                Length [did] = 0
                    
            score_doc[did] +=   IDF[t] *IDF[t]* math.log(1 + tf)
            Length [did] += (IDF[t] *IDF[t]* math.log(1+tf))**2

    for key in score_doc.keys():
        if Length[key] !=0:
            score_doc[key] /= math.sqrt(Length[key]) * math.sqrt(len_que)

    final_docs =[x for x in heapq.nlargest(10 , score_doc.items() , key=itemgetter(1))]
    return final_docs

def T2(query , IDF ,  champ_list , Doc_tf):#getting top 10 docs from for the query from the champion list

    score_doc ={}
    Length ={}
    len_que =0;
    for t in query:

        pos_list = champ_list[t]
        len_que += (IDF[t])**2
        for did in pos_list:

            if did not in score_doc.keys():
                score_doc[did] = 0
                Length [did] = 0
            
            a = [item[1] for item in Doc_tf[t] if item[0] == did ]
            tf =a[0]
                  
            score_doc[did] += IDF[t] * IDF[t] * math.log(1+tf)
            Length [did] += (IDF[t] *IDF[t]* math.log(1+tf))**2

    for key in score_doc.keys():
        if Length[key] !=0:
            score_doc[key] /= math.sqrt(Length[key]) * math.sqrt(len_que)

    final_docs = [x for x in heapq.nlargest(10 ,score_doc.items(),key=itemgetter(1))]
    return final_docs

def find_leader(query , Leaders , IDF ,Doc_tf ):#find leader doc

    score_doc ={}
    Length ={}
    len_que =0;

    for t in query:

        len_que += (IDF[t])**2
        for did in Leaders:

            if did not in score_doc.keys():
                score_doc[did] = 0
                Length [did] = 0
            
            a = [item[1] for item in Doc_tf[t] if item[0] == did ]

            if a ==[]:
                tf =0
            else:
                tf =a[0]

            score_doc[did] += IDF[t] * IDF[t] * math.log(1+tf)
            Length [did] += (IDF[t] *IDF[t]* math.log(1+tf))**2

    for key in score_doc.keys():

        if Length[key] != 0:
            score_doc[key] /= math.sqrt(Length[key]) * math.sqrt(len_que)

    lead_doc = max(score_doc, key=score_doc.get)
    
    return lead_doc

def find_followers(lead_doc , Leaders ,no_docs, vector ):#finding followers for leader doc

    
    follower = []
    score ={}
    
    for lead_id in Leaders:
        score[lead_id] = {}
    
    for doc_id in range(no_docs):   #for lead_id in Leaders:

        if doc_id not in Leaders:
            for lead_id in Leaders:

                s =1 - distance.cosine(vector[lead_id] , vector[doc_id])
                score[lead_id][doc_id] = s
               

            max_score = max((d[doc_id]) for d in score.values()) 
            

            if score[lead_doc][doc_id] >=max_score:
                follower.append(doc_id)
    
    return follower





def calc_vectors(IDF ,Doc_tf , no_docs):#computing tfidf vectors for docs

    vector ={}

    for doc_id in range(no_docs):
        for term in Doc_tf.keys():

            if doc_id not in vector.keys():
                vector[doc_id] =[]

            a = [item[1] for item in Doc_tf[term] if item[0] == doc_id ]

            if a ==[]:
                tf = 0
            else:
                tf = a[0]

            tf_idf = math.log(1+tf) * IDF[term]
            vector[doc_id] .append(tf_idf)
    #print(vector)
    return vector

def T4(query ,A ,IDF,Doc_tf):#finding top docs using cluster pruning scheme

    score_doc ={}
    Length ={}
    len_que =0;

    for t in query:

        
        len_que += (IDF[t])**2
        for did in A:

            if did not in score_doc.keys():
                score_doc[did] = 0
                Length [did] = 0
                    
            a = [item[1] for item in Doc_tf[t] if item[0] == did ]

            if a==[]:
                tf =0
            else:
                tf =a[0]

            score_doc[did] +=   IDF[t] *IDF[t]* math.log(1 + tf)
            Length [did] += (IDF[t] *IDF[t]* math.log(1+tf))**2

    for key in score_doc.keys():

        if Length[key] != 0:
            score_doc[key] /= math.sqrt(Length[key]) * math.sqrt(len_que)

    final_docs =[x for x in heapq.nlargest(10 , score_doc.items() , key=itemgetter(1))]
    return final_docs    

def answer_query(IDF , Doc_tf ,champ_list_global , champ_list_local,Leader , no_docs):#answering diff queries and storing results in file 

    
    vector = calc_vectors(IDF ,Doc_tf , no_docs)
    path = sys.argv[1]
    with open(path , 'r', encoding ="ascii", errors ="surrogateescape") as d: 
        query_file = d.read().splitlines()  

    
    file = open("RESULTS2 20CS60R44.txt" ,'a')
    for q in query_file:
           
        query = preprocessing(q)
        
        s = ""
        
        file.write(q)
        
        file.write("\n")
        final_docs = T1(query , IDF ,  Doc_tf)
        
        c = len(final_docs)
        for doc, score in final_docs:

            if score !=0:
                s += "<doc"+ str(doc) +" , " + str(score) +">"
                c-=1
                if c:
                    s +=","

        file.write(s)
        file.write("\n")
        s =""
        final_docs = T2(query , IDF ,  champ_list_local , Doc_tf)
        c = len(final_docs)
        for doc, score in final_docs:
            if score !=0:
                s +="<doc"+ str((doc)) +" , " + str(score) +">"
                c-=1
                if c:
                    s +=","
        file.write(s)

        file.write("\n")
        s =""
        final_docs = T2(query , IDF ,  champ_list_global , Doc_tf)
        c = len(final_docs)
        for doc, score in final_docs:
            if score !=0:
                s +="<doc"+ str((doc)) +" , " + str(score) +">"
                c-=1
                if c:
                    s +=","
        file.write(s)

        file.write("\n")
        lead_doc = find_leader(query , Leader , IDF ,Doc_tf )
        
        
        followers = find_followers(lead_doc , Leader ,no_docs , vector)
        A = followers + [lead_doc]

        s=""
        final_docs = T4(query ,A ,IDF,Doc_tf)
        c = len(final_docs)
        for doc, score in final_docs:
            if score !=0:
                s +="<doc"+ str((doc)) +" , " + str(score) +">"
                c-=1
                if c:
                    s +=","
        file.write(s)

        file.write("\n")
        



def main():

    
    Doc_index=0
    All_terms = set()
    filtered_Documents = {}
    Corpus= os.listdir("../Dataset/Dataset") #directory containing all text files
    no_docs = len(Corpus)
    for Doc_name in Corpus:
      
        with open("../Dataset/Dataset/" + Doc_name, 'r', encoding ="ascii", errors ="surrogateescape") as d: 
            page_html = d.read()

        soup = BeautifulSoup(page_html,'html.parser')
        filtered_Documents[Doc_index] = preprocessing (soup.get_text())
        All_terms = All_terms.union(set(filtered_Documents[Doc_index])) 
        Doc_index+=1

    print("Got the preprocessed Documents")
    All_terms = sorted(All_terms)

    
    term_doc_count,doc_tf = TF(All_terms , filtered_Documents)

    
    print("Got the document frequency for all docs")

    inv_pos_index ,IDF= idf(All_terms ,len(filtered_Documents),term_doc_count,doc_tf)
    
    

    champ_list_local = get_champ_list_local(doc_tf )
    print("Got the local champion list")
    
    

    file=open("../Dataset/StaticQualityScore.pkl", "rb") 
    gd = pickle.load(file) 
    
    champ_list_global,score_gd = get_champ_list_global(inv_pos_index, IDF ,gd)
    print("Got the global champion list")
    
    

    file=open("../Dataset/Leaders.pkl", "rb") 
    Leader = pickle.load(file)
    

    answer_query(IDF , doc_tf ,champ_list_global ,champ_list_local , Leader , no_docs)
    print("All the queries have been answered")
    
    

main()