import re


import nltk
import numpy

nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from gensim import corpora, similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import Similarity
from nltk.stem import PorterStemmer


ps = PorterStemmer()

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

import pandas as pd
import preprocessor as tp

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

tp.set_options(tp.OPT.URL,tp.OPT.MENTION)

#from imblearn.ensemble import BalancedBaggingClassifier
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import random
import copy
import argparse
import scipy.sparse as sp
import numpy as np
import time
import math
from random import randint
try: 
    import queue
except ImportError:
    import Queue as queue
    
import pickle
import os
from multiprocessing import Pool as ProcessPool
import itertools
from functools import partial



# import user pythons file
from topic_description import TRECTopics
from systemReader import systemReader
from global_definition import *

rng = np.random.seed(5)
random.seed(5)





# actual active learning for TREC is happening here for a particular topicID
# here we run for either all documents in the collection
# or all documents in the official qrels
def active_learning_multi_processing(topicId, df, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address, train_per_centage, use_pooled_budget, per_topic_budget_from_trec_qrels, feature_type):
    train_index_list = topic_seed_info[topicId]

    #topic_complete_qrels = pickle.load(open(topic_complete_qrels_address + topicId + '.pickle', 'rb'))

    original_labels = {}
    for row_index, row in df.iterrows():
        original_labels[row_index] = row['majority_label']


    original_predicted_merged_dict = {}
    original_labels_list = []
    number_of_1 = 0
    for k, v in original_labels.items():
        original_predicted_merged_dict[k] = v
        if v == 1.0:
            number_of_1 = number_of_1 + 1
        original_labels_list.append(v)
    #exit(0)

    #print "tmp_l1:",original_labels_list.count(1)


    #print "tmp_l2:",predicted_labels_list.count(1)

    #print "sum", original_labels_list.count(1) + predicted_labels_list.count(1)

    original_predicted_merged_list = []
    for k in sorted(original_predicted_merged_dict.keys()):
        #print k, original_predicted_merged_dict[k]
        original_predicted_merged_list.append(original_predicted_merged_dict[k])

    #print "again sum", original_predicted_merged_list.count(1)


    # need to convert y to np.array the Y otherwise Y[train_index_list] does not work directly on a list
    y = np.array(original_predicted_merged_list)  # 2 is complete labels of all documents in document collection
    # type needed because y is an object need and throws error Unknown label type: 'unknown'
    y = y.astype('int')
    #print "numpy sum", np.count_nonzero(y)
    #print y

    #print y.shape
    #print train_index_list
    #print y[train_index_list]

    #exit(0)

    total_documents = len(y)
    total_document_set = set(np.arange(0, total_documents, 1))

    initial_X_test = []
    test_index_dictionary = {}
    test_index_counter = 0

    #print "Starting Test Set Generation:"
    #start = time.time()
    for train_index in range(0, total_documents):
        if train_index not in train_index_list:
            initial_X_test.append(document_collection[train_index])
            test_index_dictionary[test_index_counter] = train_index
            test_index_counter = test_index_counter + 1

    #print "Finshed Building Test Set:", time.time() - start

    predictableSize = len(initial_X_test)
    isPredictable = [1] * predictableSize  # initially we will predict all

    # initializing the train_size controller
    train_size_controller = len(train_index_list)
    loopCounter = 1  # loop starts from 1 because 0 is for seed_set
    topic_all_info = {}  # key is the loopCounter

    while True:
        #print "iteration:", loopCounter
        # here modeling is utilizing the document selected in previous
        # iteration for training
        # when loopCounter == 0
        # model is utilizing all the seed document collected at the begining
        if al_classifier == 'LR':
            #model = LogisticRegression(solver=large_data_solver, C=large_data_C_parameter, max_iter=200)
            model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   n_jobs=None, penalty='l2',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)

        elif al_classifier == 'SVM':
            model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability = True)
        elif al_classifier == 'RF':
            model =  RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
        elif al_classifier == 'RFN':
            model = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=0)
        elif al_classifier == 'RFN100':
            model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
        elif al_classifier == 'NB':
            model = MultinomialNB()
        elif al_classifier == 'Ada':
            # base model is decision tree
            # logistic regression will not help
            model = AdaBoostClassifier(n_estimators=50,
                                     learning_rate=1)
        elif al_classifier == 'Xgb':
            model = XGBClassifier(random_state=1, learning_rate=0.01)
        elif al_classifier == 'BagLR':
            LRmodel = LogisticRegression(solver=large_data_solver, C=large_data_C_parameter, max_iter=200)
            model = BaggingClassifier(LRmodel, n_estimators = 5, max_samples = 1) # If float, then draw max_samples * X.shape[0] samples. 1 means use all samples
        elif al_classifier == 'BagNB':
            model = BaggingClassifier(MultinomialNB(), n_estimators = 5, max_samples = 0.5) # If float, then draw max_samples * X.shape[0] samples. 1 means use all samples
        elif al_classifier == 'Vot':
            LRmodel = LogisticRegression(solver=large_data_solver, C=large_data_C_parameter, max_iter=200)
            NBmodel = MultinomialNB()
            model = VotingClassifier(estimators=[('lr', LRmodel), ('nb', NBmodel)], voting = 'soft')

        model.fit(document_collection[train_index_list], y[train_index_list])

        #print(model.coef_)
        #print(len(model.coef_[0]))

        test_index_list = list(total_document_set - set(train_index_list))
        pooled_document_count = len(set(train_index_list).intersection(set(original_labels_list)))

        y_actual = None
        y_pred = None
        y_pred_all = []

        if isPredictable.count(1) != 0:
            y_pred = []
            for test_index_elem in test_index_list:
                if feature_type == "tfidf":
                    y_pred.append(model.predict(document_collection[test_index_elem]))
                elif feature_type == "bert" or feature_type == "robert":
                    y_pred.append(model.predict([document_collection[test_index_elem]]))

            start = time.time()
            #print 'Statred y_pred_all'
            y_actual = np.concatenate((y[train_index_list], y[test_index_list]), axis=None)
            y_pred_all = np.concatenate((y[train_index_list], y_pred), axis=None)
            '''
            for doc_index in range(0,total_documents):
                if doc_index in train_index_list:
                    y_pred_all.append(y[doc_index])
                else:
                    # result_index in test_set
                    # test_index_list is a list of doc_index
                    # test_Index_list [25, 9, 12]
                    # test_index_list[0] = 25 and its prediction in y_pred[0] --one to one mapping
                    # so find the index of doc_index in test_index_list using
                    pred_index = test_index_list.index(doc_index)
                    y_pred_all.append(y_pred[pred_index])
            '''
            #print "Finsh y_pred_all", time.time() - start

        else: # everything in trainset
            y_pred = y
            y_actual = y
            y_pred_all = y
            test_index_list = train_index_list

        f1score = f1_score(y_actual, y_pred_all, average='binary')
        precision = precision_score(y_actual, y_pred_all, average='binary')
        recall = recall_score(y_actual, y_pred_all, average='binary')

        number_of_1_found_so_far = list(y[train_index_list]).count(1)
        prevalence = (number_of_1_found_so_far*1.0)/number_of_1
        #print f1score, precision, recall, len(train_index_list), len(test_index_list), len(y_pred_all)


        # save all info using (loopCounter - 1)
        # list should be deep_copy otherwise all will point to final referecne at final iterraion
        topic_all_info[loopCounter - 1] = (topicId, f1score, precision, recall, copy.deepcopy(train_index_list), test_index_list, y_pred, pooled_document_count, prevalence)

        #print(loopCounter - 1, number_of_1_found_so_far, number_of_1, prevalence)

        # it means everything in the train list and we do not need to predict
        # so we do not need any training of the model
        # so break here
        if isPredictable.count(1) == 0:
            break
        #print isPredictable.count(1)
        # suppose original budget is 5,
        # then when train_index_list is 5, we cannot just turn off Active learning
        # we need to use that AL with train_index_list of size 5 to train use that to predict the rest
        # so we cannot exit at 5, we should exit at 5 + 1
        # that is the reason we set per_topic_budget_from_trec_qrels[topicId] + 1 where 1 is the batch_size
        # it means everything of pooled_budget size in the train_list so we need not tany training of the model
        # so break here
        if use_pooled_budget == 1 and per_topic_budget_from_trec_qrels[topicId] == len(train_index_list):
            break

        queueSize = isPredictable.count(1)
        my_queue = queue.PriorityQueue(queueSize)

        # these are used for SPL
        randomArray = []

        for counter in range(0, predictableSize):
            if isPredictable[counter] == 1:
                # model.predict returns a list of values in so we need index [0] as we
                # have only one element in the list
                y_prob = None
                if feature_type == "tfidf":
                    y_prob = model.predict_proba(initial_X_test[counter])[0]
                elif feature_type == "bert" or feature_type == "robert":
                    y_prob = model.predict_proba([initial_X_test[counter]])[0]

                #print(y_prob)
                val = 0
                if al_protocol == 'CAL':
                    val = (-1)*y_prob[1] # -1 is needed priority do sorting increasing
                    my_queue.put((val, counter))
                elif al_protocol == 'SAL':
                    val = (-1)*calculate_entropy(y_prob[0], y_prob[1])
                    my_queue.put((val, counter))
                elif al_protocol == 'SPL':
                    randomArray.append(counter)

        if use_pooled_budget == 1:
            #print "use pooled budget"
            size_limit = math.ceil(train_per_centage[loopCounter] * per_topic_budget_from_trec_qrels[topicId])
            #print("size limit:", size_limit, "total_docs:", per_topic_budget_from_trec_qrels[topicId])

        else:
            size_limit = math.ceil(train_per_centage[loopCounter] * total_documents)
            #print("size limit:", size_limit, "total_docs:", total_documents)
        if al_protocol == 'SPL':
            random.shuffle(randomArray)
            #randomArray.reverse()
            batch_counter = 0
            # for batch_counter in range(0, batch_size):
            #    if batch_counter > len(randomArray) - 1:
            #        break
            while True:
                if train_size_controller == size_limit:
                    break

                itemIndex = randomArray[batch_counter]
                isPredictable[itemIndex] = 0
                train_index_list.append(test_index_dictionary[itemIndex])
                train_size_controller = train_size_controller + 1
                batch_counter = batch_counter + 1


        else:
            while not my_queue.empty():
                if train_size_controller == size_limit:
                    break
                item = my_queue.get()
                #print(item) # is a tuple where item[1] is the index , item[0] is the predit value
                isPredictable[item[1]] = 0  # not predictable

                train_index_list.append(test_index_dictionary[item[1]])
                train_size_controller = train_size_controller + 1

        loopCounter = loopCounter + 1
    return topic_all_info

def active_learning(topic_list, df, al_protocol, al_classifier, document_collection, topic_seed_info, topic_complete_qrels_address,train_per_centage, data_path, file_name, use_pooled_budget, per_topic_budget_from_trec_qrels, feature_type):
    num_workers = None
    workers = ProcessPool(processes = 1)
    with tqdm(total=len(topic_list)) as pbar:
        partial_active_learning_multi_processing = partial(active_learning_multi_processing, df=df, al_protocol=al_protocol, al_classifier = al_classifier, document_collection=document_collection,topic_seed_info=topic_seed_info,topic_complete_qrels_address=topic_complete_qrels_address,train_per_centage=train_per_centage, use_pooled_budget=use_pooled_budget, per_topic_budget_from_trec_qrels=per_topic_budget_from_trec_qrels, feature_type=feature_type)
        for topic_all_info in tqdm(workers.imap_unordered(partial_active_learning_multi_processing, topic_list)):
            topicId = topic_all_info[0][0] # 0 is the loopCounter Index and 0 is the first tuple
            file_complete_path = data_path + file_name + str(topicId) + ".pickle"
            pickle.dump(topic_all_info, open(file_complete_path, 'wb'))
            pbar.update()



def text_preprocessing(s):
    s = str(s)
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    #s = " ".join([word for word in s.split()
    #              if word not in stopwords.words('english')
    #              or word in ['not', 'can']])
    # Remove trailing whitespace
    s = s.replace("https", "")
    s = re.sub(r'\s+', ' ', s).strip()

    s = re.compile('RT @').sub('@', s, count=1)
    s = s.replace(":", "").strip()
    s = tp.clean(s)

    tokens = nltk.word_tokenize(s)
    stems = []
    for item in tokens:
        stems.append(ps.stem(item))

    #print(s)

    #print(stems)
    return stems




#########################################################################
# main script
# create all directories manually before running otherwise multi-processing will create lock condition
# for creating files

if __name__ == '__main__':


    # read the processed csv file in a dataframe

    input_file = "/home/nahid/Downloads/data_all/version2/mturk/processed/only_raw_manual_agrees.csv"
    hate_df = pd.read_csv(input_file)


    classifier_name = "LR"
    classifier = None
    if classifier_name == 'LR':
        classifier = LogisticRegression(solver=small_data_solver,C=small_data_C_parameter)



    # only IS and, RDS
    topicData = TRECTopics("Hate", 1, 2)
    seed_selection_type = "IS"
    topic_seed_info_file_name = "per_topic_seed_documents_" + seed_selection_type
    number_of_seeds = 10
    data_path = "/home/nahid/Downloads/data_all/version2/AL/"
    file_name = "seed_dcouments"
    seed_selection_type = "IS"
    topic_seed_info = topicData.get_seed_tweets(hate_df, number_of_seeds, data_path, file_name, seed_selection_type)

    print(topic_seed_info)

    '''
    # sanity check
    for topicId in sorted(topic_seed_info.iterkeys()):
       #print topicId, len(topic_seed_info[topicId]), topic_seed_info[topicId]
       print topicId, len(topic_seed_info[topicId])
    '''

    feature_type_list = ["tfidf", "robert"]
    al_protocol_list = ['CAL', 'SAL', 'SPL']

    #v = TfidfVectorizer(analyzer=text_preprocessing, min_df=1, max_features=4500)
    v = TfidfVectorizer(analyzer=text_preprocessing, ngram_range=(1, 3), binary=True, smooth_idf=False)

    document_collection = v.fit_transform(hate_df['tweet'])

    start = time.time()
    # topic_list = [str(topicID) for topicID in range(start_topic[datasource], end_topic[datasource])]
    start_top = 1
    end_top = 2
    use_pooled_budget = 0
    #al_classifier = "LR"
    al_classifier = classifier_name
    per_topic_budget_from_trec_qrels = None

    topic_list = [str(topicID) for topicID in range(start_top, end_top)]
    print("topic_list", topic_list)

    loaded_bert_features = pickle.load(open("/home/nahid/Downloads/data_all/mturk/processed/bert_feature.pickle", "rb"))

    loaded_robert_features = pickle.load(open("/home/nahid/Downloads/data_all/version2/mturk/processed/robert_feature.pickle", "rb"))
    loaded_robert_features = numpy.array(loaded_robert_features)

    print(loaded_robert_features.shape)
    print(len(loaded_robert_features))
    print(len(loaded_robert_features[0]))
    print(loaded_robert_features[0])


    for feature_type in feature_type_list:
        if feature_type == "bert":
            document_collection = loaded_bert_features
        elif feature_type == "robert":
            document_collection = loaded_robert_features
        for al_protocol in al_protocol_list:
            print(al_protocol)

            classifier_name = "LR"
            classifier = None
            if classifier_name == 'LR':
                classifier = LogisticRegression(solver=small_data_solver, C=small_data_C_parameter)

            topic_complete_qrels_address = data_path + "per_topic_complete_qrels_" + classifier_name + "_"
            topic_all_info_file_name = "per_topic_predictions_" + seed_selection_type + "_" + classifier_name + "_" + al_protocol + "_" + feature_type + "_"
            # print data_path + topic_all_info_file_name + topic_list[0] + '.pickle'
            topic_all_info = active_learning(topic_list, hate_df, al_protocol, al_classifier, document_collection,
                                             topic_seed_info, topic_complete_qrels_address,
                                             train_per_centage, data_path, topic_all_info_file_name,
                                             use_pooled_budget, per_topic_budget_from_trec_qrels, feature_type)

            #print(v.get_feature_names())
            #print(len(v.get_feature_names()))
            #print(len(document_collection))



            # sanity check
            for topicId in topic_list:
                print("topicId:", topicId)

                topic_all_info = pickle.load(open(data_path+ topic_all_info_file_name+ str(topicId) + ".pickle",'rb'))
                for k in sorted(topic_all_info.keys()):
                    #(0      ,       1,         2,      3,                              4,               5,   6
                    # topicId, f1score, precision, recall, copy.deepcopy(train_index_list), test_index_list, y_pred, pooled_document_count)

                    print (k*10, topic_all_info[k][0], topic_all_info[k][1], topic_all_info[k][3], len(topic_all_info[k][4]), len(topic_all_info[k][5]), len(topic_all_info[k][6]))

    '''
    datasource = sys.argv[1]  # can be 'TREC8','gov2', 'WT2013','WT2014'
    al_protocol = sys.argv[2]  # 'SAL', 'CAL', # SPL is not there yet
    seed_selection_type = sys.argv[3]  # 'IS' only
    classifier_name = sys.argv[4]  # "LR", "NR"--> means non-relevant all
    collection_size = sys.argv[5]  # 'all', 'qrels' qrels --> means consider documents inseide qrels only
    al_classifier = sys.argv[6]  # SVM, RF, NB and LR
    start_top = int(sys.argv[7])
    use_pooled_budget = int(sys.argv[8])  # 1 means use and 0 does not use that
    use_original_qrels = int(sys.argv[9])  # 1 means use original qrels, other value 0 means
    varied_qrels_directory_number = int(sys.argv[10])  # 1,2,3
    end_top = start_top + 1
    '''
