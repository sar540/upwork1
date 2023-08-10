

#S_lightning_classifer_ML_model_aug7_2023
#simple model without hyperparamters search 
'''
Python 3.10.11
import  sklearn
sklearn.__version__
'1.3.0'

'''
import sys
print('python version is ', sys.version)
print('path to python exe ' ,  sys.executable)
import  sklearn
print('sklearn  version' , sklearn.__version__)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

import  numpy as  np



if 0: #  use only 0
    q = 0
else: #read data from file 
    q = 0
    text_file = open("y_test.txt", "r")
    y_test = text_file.readlines()    
    text_file = open("y_train.txt", "r")
    y_train = text_file.readlines()
    y_train =  [int(x) for x in y_train]
    text_file = open("X_test.txt", "r")
    X_test = text_file.readlines()    
    text_file = open("X_train.txt", "r")
    X_train = text_file.readlines()
    q = 0

#preprocessing 
vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)
X_train_vectorised = vectorizer.fit_transform(X_train)
q = 0

if 1:
    #https://github.com/scikit-learn-contrib/lightning/blob/master/examples/document_classification_news20.py
    from lightning.classification import CDClassifier #Estimator for learning linear classifiers by (block) coordinate descent
    #import  lightning
    clf_lightning = CDClassifier(loss="squared_hinge",
                       #penalty="l2",
                       penalty="l1",
        #penalty="l1/l2", #                     
                       multiclass=False,
                       max_iter=20,
                       alpha=1e-4,
                       C=1.0 / X_train_vectorised.shape[0],
                       tol=1e-3,
                       n_jobs =5)
    #
    #https://contrib.scikit-learn.org/lightning/generated/lightning.regression.FistaRegressor.html
    #clf =  lightning.regression.FistaRegressor(C=1.0, alpha=1.0, penalty='l1', max_iter=100, max_steps=30, eta=2.0, sigma=1e-05, callback=None, verbose=0)
    clf_lightning.fit(X_train_vectorised, np.array(y_train))
    pred_train_lightning = [int(x) for x in clf_lightning.predict(X_train_vectorised)]   
    original_targt_lightning = [int(x) for x in y_train]  
    print('****                 lightning confusion_matrix       **************')
    print(confusion_matrix(pred_train_lightning, original_targt_lightning)  )
    print('****                 lightning confusion_matrix       **************')
    print('\nlightning score')
    print(clf_lightning.score(X_train_vectorised, y_train))
    print(clf_lightning.n_nonzero(percentage=True))
    q = 0
    
    



#  input your text 
my_text =  X_test[0]
#my_text =  'From: teezee@netcom.com (TAMOOR A. ZAIDI)\nSubject: Hall Generators from USSR\nKeywords: hall generators,thrusters,USSR,JPL\nOrganization: NETCOM On-line Communication Services (408 241-9760 guest)\nLines: 21\n\nHi Folks,\n\n              Last year America bought two  "Hall Generators" which are\nused as thrusters for space vehicles from former USSR,if I could recall\ncorrectly these devices were sent to JPL,Pasadena labs for testing and\nevaluation.\n     \n              I am just curious to know  how these devices work and what\nwhat principle is involved .what became of them.There was also some\ncontroversy that the Russian actually cheated,sold inferior devices and\nnot the one they use in there space vehicles.\n\nAny info will be appreciated...\n  ok   {                         Thank{ in advance...\nTamoor A Zaidi\nLockheed Commercial Aircraft Center\nNorton AFB,San Bernardino\n\nteezee@netcom.com\nde244@cleveland.freenet.edu\n\n'
#my_text =  'Mac and was wondering if anyone in netland knows of public domain anti-aliasing utilities so that I can skip this step '

#  predict your text
#preprocessing
test_vectorised  =  vectorizer.transform([my_text])
#predict use existing model 

pred_test_lightning      =  clf_lightning.predict(test_vectorised)

print('for text => ', my_text )

print('pred_test_lightning', pred_test_lightning[0]) 
q = 0