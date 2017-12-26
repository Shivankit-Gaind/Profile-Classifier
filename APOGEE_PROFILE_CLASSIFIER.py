import slate
import numpy as np

#####################################################################################################################

#making a list of resumes

lisa = []

for i in range(36):
    filename="\c"+str(i+1) +".pdf"
    f=open("CVs\Selected\cs"+filename,"rb")
    doc = slate.PDF(f)
    randomstring=""	
    for j in range(len(doc)):
        randomstring +=doc[j]
    lisa.append(randomstring)
	
	
for i in range(29):
    filename="\ec"+str(i+1) +".pdf"
    f=open("CVs\Not_selected"+filename,"rb")
    doc = slate.PDF(f)
    randomstring=""
    for j in range(len(doc)):
        randomstring +=doc[j]
    lisa.append(randomstring)
	
	
for i in range(32):
    filename="\hss"+str(i+1) +".pdf"
    f=open("CVs\Not_selected"+filename,"rb")
    doc = slate.PDF(f)
    randomstring=""
    for j in range(len(doc)):
        randomstring +=doc[j]
    lisa.append(randomstring)
	
######################################################################################################################
	
# removing punctuation and other stuff

import string
for i in range(len(lisa)):
    lisa[i] = lisa[i].translate(None,string.punctuation)
    lisa[i] = lisa[i].translate(None,"\n")
import re
for i in range(len(lisa)):
    lisa[i]=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]','', lisa[i])
	
#######################################################################################################################	

#labels for the resumes

labels = []
for i in range(36):
    labels.append(1)
for i in range(61):
    labels.append(0)
labels = np.array(labels)

#######################################################################################################################


#shuffling data and splitting it up for training and testing

from sklearn.model_selection import train_test_split
features_train,features_test,y_train,y_test=train_test_split(lisa,labels,test_size=0.33,random_state=42)

#######################################################################################################################

# preparing a feature matrix from the resumes (training data)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer="word",stop_words="english",max_features=250)
words = vectorizer.fit_transform(features_train)
X_train = words.toarray()

X_test = vectorizer.transform(features_test)
X_test = X_test.toarray()

#######################################################################################################################

#applying Decision Tree Classifier 

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
print clf.score(X_train,y_train)
print clf.score(X_test,y_test)

#######################################################################################################################

#applying Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier()
clf1=clf1.fit(X_train,y_train)
print clf1.score(X_train,y_train)
print clf1.score(X_test,y_test)

#######################################################################################################################

#applying SVM Classifier

from sklearn import svm
model_svm=svm.SVC()
model_svm=model_svm.fit(X_train,y_train)
print model_svm.score(X_train,y_train)
print model_svm.score(X_test,y_test)

#######################################################################################################################

#appling Naive Bayes Classifier

from sklearn.naive_bayes import BernoulliNB
clf4 = BernoulliNB()
clf4 = clf4.fit(X_train,y_train)
print clf4.score(X_train,y_train)
print clf4.score(X_test,y_test)

#######################################################################################################################








