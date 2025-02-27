from collections import Counter
import math
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

corpusroot = './US_Inaugural_Addresses'
documents_length = 0
df = Counter()
tf={}
document_vector = {}
cosine_sims = Counter()  
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()
        # Tokenize the document
        tokens = tokenizer.tokenize(doc)

        #remove stop word
        tokens = [word for word in tokens if word not in stop_words] 
   
        #Porter Stemming
        tokens = [stemmer.stem(word) for word in tokens]
        #store the value
        df = df + Counter(list(set(tokens)))
        tf[filename] = Counter(tokens)

def getidf(token,stem="yes"):
    if stem == "yes":
        token = stemmer.stem(token)
    else:
        token = token
    if df[token] != 0:
        return (math.log10(len(tf)/(df[token])))
    else:
        return -1
    
def compute_idf_for_doc(documents):
    idf = {}
    N = len(documents)
    for filename, tokens in documents.items():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            idf[token] = idf.get(token, 0) + 1
    for token, count in idf.items():
        idf[token] = math.log10(N / count)
    return idf    
idf = compute_idf_for_doc(tf) 
#normalize the weight vector
def normalizeValue(weight_vector):
    normalized_vector={}

    doc_length = 0
    for i in weight_vector:
        doc_length += weight_vector[i]**2
        
        #calculating the magnitude   
    doc_magnitude = math.sqrt(doc_length)
    for i in weight_vector:
        normalized_vector[i] = weight_vector[i]/doc_magnitude
    return normalized_vector 

#calculate the tf-idf weight for each term in the document
def get_tf_idf_weight_document(filename,token):
    if(tf[filename][token])!=0: # checking if the term is present in the document or not
        return (1+math.log10(tf[filename][token]))*getidf(token,"no")
    else:
        return 0
    
#tf-idf weight for each term in the document
for doc in tf:
    weight_vector = Counter()
    for word in tf[doc]:
        weight_vector[word] = get_tf_idf_weight_document(doc,word)  
    document_vector[doc]  = normalizeValue(weight_vector) 

#sort the dictionary by value in descending order
for filename in document_vector:
    sorted_dict = dict(sorted(document_vector[filename].items(), key=lambda x: x[1], reverse=True))
    document_vector[filename] = sorted_dict 
    
def getweight(filename, token):
    stemmed_token = stemmer.stem(token)
    if filename in document_vector and stemmed_token in document_vector[filename]:
        return document_vector[filename][stemmed_token]
    else:
        return 0
            
def query(qstring):
    #convert to lower case
    qstring = qstring.lower()
    #tokenize the query string
    tokens = tokenizer.tokenize(qstring)
    #remove stop word
    
    q_token = [word for word in tokens if word not in stop_words] 
    #Porter Stemming
    q_tokens = [stemmer.stem(token) for token in q_token]
    if not q_tokens:
        return ("None", 0)
    q_tfidf = {}
    q_token = Counter(q_tokens)
    for token in q_token:
        q_tfidf[token] = (1 + math.log(q_token[token],10)) * getidf(token,"No")
    #calculate the magnitude of the query vector
    query_magnitude = math.sqrt(sum([i**2 for i in q_tfidf.values()]))
    
    for token in q_tfidf:
        q_tfidf[token] = q_tfidf[token]/query_magnitude
        
    #calculate the cosine similarity
    postings = {}
    for token in q_tokens:
        if token in idf:
            postings[token] = sorted([(filename, document_vector[filename].get(token, 0)) for filename in tf], key=lambda x: -x[1])[:10]
    
    # Collect all documents that appear in the top 10 of at least one query token
    candidate_docs = set()
    for token in q_tokens:
        if token in postings:
            candidate_docs.update([doc for doc, _ in postings[token]])
    
    # If no candidate documents, return ("None", 0)
    if not candidate_docs:
        return ("None", 0)
    
    # Calculate actual scores and upper-bound scores for candidate documents
    actual_scores = {}
    upper_bound_scores = {}
    
    for doc in candidate_docs:
        actual_score = 0
        upper_bound_score = 0
        
        for token in q_tokens:
            if token in postings:
                # Check if the document is in the top 10 for this token
                doc_in_top_10 = any(doc == d for d, _ in postings[token])
                
                if doc_in_top_10:
                    # Use actual weight
                    actual_score += q_tfidf[token] * document_vector[doc].get(token, 0)
                    upper_bound_score += q_tfidf[token] * document_vector[doc].get(token, 0)
                else:
                    # Use upper-bound weight (10th element's weight)
                    if len(postings[token]) >= 10:
                        upper_bound_weight = postings[token][9][1]  # 10th element's weight
                        upper_bound_score += q_tfidf[token] * upper_bound_weight
        
        actual_scores[doc] = actual_score
        upper_bound_scores[doc] = upper_bound_score
    
    # Find the document with the highest actual score
    best_doc = max(actual_scores, key=actual_scores.get)
    best_score = actual_scores[best_doc]
    
    # Check if the best document's actual score is better than or equal to all other documents' upper-bound scores
    is_best = True
    for doc in candidate_docs:
        if doc != best_doc and best_score < upper_bound_scores[doc]:
            is_best = False
            break
    
    if is_best:
        return (best_doc, best_score,"GUARANTEED WINNER")
    else:
        # If no document satisfies the condition, return ("fetch more", 0)
        return ("fetch more", 0)

#query1 - calculating IDF

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('dollar'))
print("%.12f" % getidf('constitution'))
print("%.12f" % getidf('power'))

#query2 - calculating weights
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','states'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('05_jefferson_1805.txt','false'))
print("%.12f" % getweight('22_grant_1873.txt','proposition'))
print("%.12f" % getweight('16_taylor_1849.txt','duties'))

#query3 - find query doc similarity
print("--------------")
print("(%s, %.12f, %s)" % query("executive power"))
print("(%s, %.12f, %s)" % query("foreign government"))
print("(%s, %.12f, %s)" % query("public rights"))
print("(%s, %.12f, %s)" % query("people government"))
print("(%s, %.12f, %s)" % query("states laws"))
