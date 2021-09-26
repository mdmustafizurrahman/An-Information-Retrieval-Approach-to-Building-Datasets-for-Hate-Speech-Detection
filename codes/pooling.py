import os
import re
#from tqdm import tqdm
import numpy as np
import pandas as pd
import preprocessor as tp

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

tp.set_options(tp.OPT.URL,tp.OPT.MENTION)
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC

import nltk
# Uncomment to download "stopwords"
#nltk.download("stopwords")
from nltk.corpus import stopwords
import copy
#import torch
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
import pickle
import json
import jsonlines
import operator
from collections import OrderedDict

seed = 2020
# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
learning_rate = 2e-5
# we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
number_of_epochs = 1

'''
dataset_name_list = ['HateEval_spanish']
train_set_list = ['HateEval_spanish']
test_set_list = ['test_tweet_spanish']
'''

dataset_name_list = ['WH', 'HateEval', 'davidson', 'goldback', 'Founta_2018', 'grimmer']
train_set_list = ['WH','davidson','HateEval', 'goldback','Founta_2018', 'grimmer']
test_set_list = ['test_tweet']

classifier_list = ['NB', 'LR']
#classifier_list = ['LR']


dataset_location = {}
dataset_location['WH'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/Waseem_and_Hovy_2016/data.csv"
dataset_location['HateEval'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/SemEval-2019/hateval2019_en_train.csv"
dataset_location['davidson'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/davidson_2017/labeled_data.csv"
dataset_location['goldback'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/Golbeck_2017/onlineHarassmentDataset.tdf"
dataset_location['rezvan'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/Rezvan_2018/final.csv"
dataset_location['grimmer'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/Grimminger_2020/combined_train_test.tsv"

dataset_location['HateEval_spanish'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/SemEval-2019/hateval2019_es_train.csv"
dataset_location['Founta_2018'] = "/work2/04549/mustaf/lonestar/data/Hate_Speech_data/Founta_2018/hatespeech_text_label_vote.csv"

#dataset_location['test_tweet'] = "/scratch/07807/dinesh/Tweet-Filtering/English-Tweets/2017-10-01-08-00-01.json"
#dataset_location['test_tweet'] = "2017-10-01-08-00-01.json"

# the following two address contains the same 49 tweets json file for 1st unterface where we have collected 4668 tweets annottaion
#dataset_location['test_tweet'] = "/work2/04549/mustaf/maverick2/twitter_corpus/English-Tweets/"
#dataset_location['test_tweet'] = "/work2/07807/dinesh/stampede2/English-Tweets/PreviousAnalyzedFile/"

# this is the new 100 tweets json file we are doring for 2nd interface, we will collect 5,000 tweet for annotation
# contains 13674964 tweets
dataset_location['test_tweet'] = "/work2/07807/dinesh/stampede2/English-Tweets/CurrentAnalyzedFile/"
dataset_location['test_tweet_spanish'] = "/work2/04549/mustaf/maverick2/spanish-tweets/2017-10-01-08-00-01.json"


dataset_separator = {}
dataset_separator['WH'] = "\t"
dataset_separator['HateEval'] = ','
dataset_separator['HateEval_spanish'] = ','

dataset_separator['davidson'] = ','
dataset_separator['goldback'] = '\t'
dataset_separator['rezvan'] = ','
dataset_separator['Founta_2018'] = '\t'
dataset_separator['grimmer'] = '\t'



def text_preprocessing(s):
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
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def read_data(dataset_name, file_address):
    test_tweets_list = []
    test_tweets_raw = []

    if dataset_name == "WH":

        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name], header= None, encoding='utf-8')
        df.columns = ["id", "text", "label"]

        df = df.drop(columns=['id'])

        df.loc[(df.label == 'none'), 'label'] = 0
        df.loc[(df.label == 'none '), 'label'] = 0

        df.loc[(df.label == 'sexism'), 'label'] = 1
        df.loc[(df.label == 'racism'), 'label'] = 1

    elif dataset_name == "HateEval":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name])
        #print(df.columns)
        df = df.drop(columns=['id','TR','AG'])
        #print(df.columns)
        df = df.rename(columns={"HS": "label"})
        #print(df.columns)


    elif dataset_name == "HateEval_spanish":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name])
        # print(df.columns)
        df = df.drop(columns=['id', 'TR', 'AG'])
        # print(df.columns)
        df = df.rename(columns={"HS": "label"})
        # print(df.columns)



    elif dataset_name == "davidson":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name])
        df = df.drop(columns=['index','count','hate_speech','offensive_language','neither'])
        df = df.rename(columns = {"class": "label", "tweet": "text"})

        #print(df.columns)

        df.loc[(df.label == 0), 'label'] = 1
        df.loc[(df.label == 2), 'label'] = 0

        #print(df.sample(20))
    elif dataset_name == "goldback":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name], encoding = "ISO-8859-1")
        df = df.drop(columns=['ID'])
        df = df.rename(columns={"Code": "label", "Tweet": "text"})

        df.loc[(df.label == 'H'), 'label'] = 1
        df.loc[(df.label == 'N'), 'label'] = 0

        #print(df.sample(10))
    elif dataset_name == "rezvan":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name])
        df.columns = ["text", "label"]

        df.loc[(df.label == 'yes'), 'label'] = 1
        df.loc[(df.label == 'Yes'), 'label'] = 1
        df.loc[(df.label == 'YES'), 'label'] = 1
        df.loc[(df.label == 'yes '), 'label'] = 1

        df.loc[(df.label == 'no'), 'label'] = 0
        df.loc[(df.label == 'No'), 'label'] = 0
        df.loc[(df.label == 'NO'), 'label'] = 0
        df.loc[(df.label == 'N'), 'label'] = 0

        df.loc[(df.label == 'Other'), 'label'] = 0
        df.loc[(df.label == 'others'), 'label'] = 0
        df.loc[(df.label == 'Others'), 'label'] = 0

        df.loc[(df.label == 'racism'), 'label'] = 1
        df.loc[(df.label == 'not sure'), 'label'] = 0

        df.loc[(df.label == 'Not Sure'), 'label'] = 0
        df.loc[(df.label == 'Not sure'), 'label'] = 0

        df.dropna(inplace = True)

    elif dataset_name == "Founta_2018":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name])
        df.columns = ["text", "label", "count"]
        df = df.drop(columns=['count'])

        df.loc[(df.label == 'normal'), 'label'] = 0
        df.loc[(df.label == 'abusive'), 'label'] = 0
        df.loc[(df.label == 'spam'), 'label'] = 0
        df.loc[(df.label == 'offensive'), 'label'] = 0
        df.loc[(df.label == 'cyberbullying'), 'label'] = 0
        df.loc[(df.label == 'Aggressive'), 'label'] = 0
        df.loc[(df.label == 'hateful'), 'label'] = 1

    elif dataset_name == "grimmer":
        df = pd.read_csv(dataset_location[dataset_name], sep=dataset_separator[dataset_name])
        #df.columns = ["text", "label", "count"]
        # text	Trump	Biden	West	HOF
        df = df.drop(columns=['Trump', 'Biden', 'West'])
        df.columns = ["text", "label"]

        print(df.label.value_counts())

        df.loc[(df.label == 'Non-Hateful'), 'label'] = 0
        df.loc[(df.label == 'Hateful'), 'label'] = 1


    elif dataset_name == "test_tweet" or dataset_name == "test_tweet_spanish":

        input_test_file_name = os.path.basename(file_address)
        input_test_file_name = input_test_file_name[:input_test_file_name.index(".json")]
        print(input_test_file_name)
        input_test_file_base_address = dataset_location[dataset_name]
        print(input_test_file_base_address)
        file_name_preprocessed = os.path.join(input_test_file_base_address, input_test_file_name + "_preprocessed.pickle")
        file_name_raw = os.path.join(input_test_file_base_address, input_test_file_name + "_raw.pickle")

        #file_name_preprocessed = dataset_name+"_preprocessed.pickle"
        #file_name_raw = dataset_name + "_raw.pickle"

        if os.path.exists(file_name_preprocessed) and os.path.exists(file_name_raw):
            test_tweets_list = pickle.load(open(file_name_preprocessed, "rb"))
            test_tweets_raw = pickle.load(open(file_name_raw, "rb"))
            return test_tweets_list, test_tweets_raw
        else:
            all_tweet_content = []
            with open(file_address) as f:
                for line in f:
                    data = json.loads(line)
                    text = data['text']
                    twitter_id = data['id']
                    #test_tweets_raw.append(text)
                    #test_tweets_list.append(text_preprocessing(text))
                    all_tweet_content.append([twitter_id, text_preprocessing(text), text])
            #pickle.dump(test_tweets_list, open(file_name_preprocessed, "wb"))

            #pickle.dump(test_tweets_raw, open(file_name_raw, "wb"))

            df = pd.DataFrame(all_tweet_content, columns=['twitter_id', 'processed_text', 'text'])

            return df


    X = df.text.values

    y = df.label.values
    y = y.astype(int)

    return df, X, y

def train_test_seprataion(data):
    X = data.text.values
    y = data.label.values

    y = y.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)
    return X_train, X_val, y_train, y_val


def tfidf_vector_generation(dataset_name):
    X_train, X_val, y_train, y_val = train_test_seprataion(read_data(dataset_name)[0], None)
    # Preprocess text
    X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
    X_val_preprocessed = np.array([text_preprocessing(text) for text in X_val])

    # Calculate TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                             binary=True,
                             smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
    X_val_tfidf = tf_idf.transform(X_val_preprocessed)

    return X_train_tfidf, X_val_tfidf, y_train, y_val



def f1_m(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary')


def compute_and_save_tfidf_vectorizer(train_dataset):

    tfidf_feature_file_name = train_dataset + "_" + "tf_idf_features.pickle"
    tfidf_model_file_name = train_dataset + "_" + "tf_idf_model.pickle"
    data_x_y_file_name = train_dataset + "_" + "data_x_y.pickle"

    if os.path.exists(tfidf_feature_file_name) and os.path.exists(tfidf_model_file_name) and os.path.exists(data_x_y_file_name):
        tf_idf_model = pickle.load(open(tfidf_model_file_name, "rb"))
        tf_idf_features = pickle.load(open(tfidf_feature_file_name, "rb"))
        data_x_y = pickle.load(open(data_x_y_file_name, "rb"))
        train_X = data_x_y[0]
        train_y = data_x_y[1]

        return tf_idf_model, tf_idf_features, train_X, train_y

    else:
        _, train_X, train_y = read_data(train_dataset, None)

        train_X_preprocessed = np.array([text_preprocessing(text) for text in train_X])

        tf_idf_model = TfidfVectorizer(ngram_range=(1, 3),
                                 binary=True,
                                 smooth_idf=False)
        tf_idf_features = tf_idf_model.fit_transform(train_X_preprocessed)

        pickle.dump(tf_idf_features, open(tfidf_feature_file_name, "wb"))
        pickle.dump(tf_idf_model, open(tfidf_model_file_name, "wb"))

        data_x_y = (train_X, train_y)
        pickle.dump(data_x_y, open(data_x_y_file_name, "wb"))

        return tf_idf_model, tf_idf_features, train_X, train_y


def train_and_save_model(classifier, train_dataset):

    trained_model_file_name = classifier + "_" + train_dataset + "_.pickle"
    if os.path.exists(trained_model_file_name):
        model = pickle.load(open(trained_model_file_name, "rb"))
        return model
    else:
        model = None
        if classifier == "NB":
            model = MultinomialNB()
        elif classifier == "LR":
            model = LogisticRegression()
        elif classifier == "GB":
            model = GradientBoostingClassifier(random_state=seed)
        if classifier == "rbf_svm":
            model = SVC(probability= True, C=0.01, kernel="rbf")  # default rbf
        elif classifier == "svm_linear":
            model = LinearSVC(C=0.01)

        tf_idf_model, tf_idf_features, train_X, train_y = compute_and_save_tfidf_vectorizer(train_dataset)
        model.fit(tf_idf_features, train_y)

        pickle.dump(model, open(trained_model_file_name, "wb"))
        return model

def get_ranked_list_of_tweet_ids(y_preds): # y_pred (pred[0], pred[1])

    id = 0
    id_prediction = {}

    for pred in y_preds:
        id_prediction[id] = pred[1]
        id = id + 1

    sorted_ids = sorted(id_prediction, key=id_prediction.get, reverse=True)  # sort tweetid by the predicted score on hate class (1) in descending order

    return sorted_ids


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# pool classifier, if we have M dataset and N classifiers
# then in total we M*N classifier to train
# we pool using M*N classifiers
if __name__ == "__main__":

    #read_data('test_tweet')
    #exit(0)

    prediction_is_done = 1 # 0 no prediction is done
    pool_start = 9000
    pool_end = 20000
    output_file_base_name = "/work2/07807/dinesh/stampede2/English-Tweets/CurrentAnalyzedFile/pooled_tweets/pool_depth_"
    data_tf_idf_models = {} # key is the dataset_name value is tf_idf_model
    data_tf_idf_features = {} # key is the dataset_name value is tf_idf_model
    data_x_y = {} # key is the dataset_name and value is the tuple (train_X, train_y)
    data_trained_model = {} # key is the dataset_name_classifier and value is the traind model


    # generate M*N trained_model
    for dataset_name in dataset_name_list:
        if dataset_name == "test_tweet":
            continue
        tf_idf_model, tf_idf_features, train_X, train_y = compute_and_save_tfidf_vectorizer(dataset_name)
        data_tf_idf_models[dataset_name] = tf_idf_model
        data_tf_idf_features[dataset_name] = tf_idf_features
        data_x_y[dataset_name] = (train_X, train_y)

        for classifier in classifier_list:
            print(dataset_name, classifier)
            trained_model = train_and_save_model(classifier, dataset_name)
            data_trained_model[dataset_name+"_"+classifier] = trained_model

    ranked_tweets_ids = {} # key dataset_name_classifier # value is the tweets id sorted by sorted by class=1 socre
    test_dataset = test_set_list[0]

    # run this part for new predcition on tests file

    if prediction_is_done == 0:
        data_frame_list = []
        for test_file in os.listdir(dataset_location[test_dataset]):
            if ".json" not in test_file:
                continue
            test_file_name = test_file[:test_file.index(".json")]
            test_file_location = os.path.join(dataset_location[test_dataset], test_file)
            if os.path.getsize(test_file_location) == 0:
                # skipping because some json file contains nothing
                continue
            print(test_file_location)
            test_X = []
            test_X_raw = []
            test_y = []

            test_df = read_data(test_dataset, test_file_location) # returing preprocess tweets
            test_X = test_df['processed_text']
            for dataset_name in train_set_list:
                for classifier in classifier_list:

                    print(test_file_name, dataset_name, classifier)

                    tf_idf_model = data_tf_idf_models[dataset_name]
                    trained_model = data_trained_model[dataset_name+"_"+classifier]
                    test_tf_idf_features = tf_idf_model.transform(test_X)
                    y_pred_probs = trained_model.predict_proba(test_tf_idf_features)
                    hate_class_probabilty = []
                    for probability in y_pred_probs:
                        hate_class_probabilty.append(probability[1])

                    test_df['prediction'] = hate_class_probabilty
                    final_df = test_df.drop(columns=['processed_text'])

                    pickle.dump(final_df, open(os.path.join(dataset_location[test_dataset], dataset_name + "_" + classifier + "_" + test_file_name + ".df"), "wb"))
                    #print(final_df.head())
                    data_frame_list.append(copy.deepcopy(final_df))
                    final_df = None

    else:
        # for prediction on new test data
        print("POOLING and DUMPING")
        for pool_depth in range(pool_start, pool_end + 100, 500):

            data_frame_list = []
            for test_file in os.listdir(dataset_location[test_dataset]):
                if ".json" not in test_file:
                    continue

                test_file_location = os.path.join(dataset_location[test_dataset], test_file)

                if os.path.getsize(test_file_location) == 0:
                    # skipping because some json file contains nothing
                    continue

                test_file_name = test_file[:test_file.index(".json")]

                for dataset_name in train_set_list:
                    for classifier in classifier_list:
                        temp_df = pickle.load(open(os.path.join(dataset_location[test_dataset], dataset_name + "_" + classifier + "_" + test_file_name + ".df"),"rb"))
                        temp_df = temp_df.sort_values('prediction',ascending = False)
                        #print(temp_df)
                        temp_df = temp_df.head(pool_depth)
                        #print(temp_df)

                        data_frame_list.append(copy.deepcopy(temp_df))
                        temp_df = None

            combined_df = pd.concat(data_frame_list)

            combined_df = combined_df.sort_values('prediction',ascending = False)
            filtered_df = combined_df[combined_df['prediction'] >= 0.9]

            # randomly sample 5000 tweets from  dataframe
            filtered_df = filtered_df.sample(n=5000)

            #print(filtered_df)
            unique_tweets = filtered_df.text.unique()
            print(pool_depth, len(unique_tweets))



            final_tweets = []
            for pooled_tweet in unique_tweets:
                raw_tweets_without_RT = re.compile('RT @').sub('@', pooled_tweet, count=1)
                raw_tweets_without_RT = raw_tweets_without_RT.replace(":", "").strip()
                cleaned_tweet = tp.clean(raw_tweets_without_RT)
                without_emoji_tweet = remove_emoji(cleaned_tweet)

                if len(without_emoji_tweet.split(" ")) >= 3:
                    final_tweets.append(cleaned_tweet)

            final_tweets = list(set(final_tweets))
            output_file_name = output_file_base_name + str(pool_depth) + "_tweets_" + str(
                len(final_tweets)) + "_pooled_tweets_filtered.text"
            print(output_file_name)

            with jsonlines.open(output_file_name, mode='w') as writer:
                for pooled_tweet in final_tweets:
                    manifest_str = {}
                    manifest_str['source'] = pooled_tweet

                    writer.write(manifest_str)





    exit(0)
    #for pool_depth in range(10, len(test_X)+1000, 10):
    for pool_depth in range(10, 1000, 10):

        pooled_tmp_tweets_id = []
        for k, list_of_tweetids in ranked_tweets_ids.items():
            #print(k, list_of_tweetids)
            pooled_tmp_tweets_id = pooled_tmp_tweets_id + list_of_tweetids[:pool_depth]

        pooled_tweet_ids = list(set(pooled_tmp_tweets_id))
        pooled_cost = len(pooled_tweet_ids)

        if test_dataset == "test_tweet" or test_dataset == "test_tweet_spanish":
            print(pool_depth, pooled_cost)
            pooled_tweets = []
            for pooled_tweet_id in pooled_tweet_ids:
                raw_tweets_without_RT = re.compile('RT @').sub('@', test_X_raw[pooled_tweet_id], count=1)
                raw_tweets_without_RT = raw_tweets_without_RT.replace(":","").strip()
                pooled_tweets.append(tp.clean(raw_tweets_without_RT))
            pickle.dump(pooled_tweets, open(test_dataset+"_"+str(pool_depth)+"_pooled_tweets.pickle", "wb"))


            with jsonlines.open(test_dataset+"_"+str(pool_depth)+"_pooled_tweets.text", mode='w') as writer:
                for pooled_tweet in pooled_tweets:
                    manifest_str = {}
                    manifest_str['source'] = pooled_tweet

                    writer.write(manifest_str)


            continue

        total_hate = np.count_nonzero(np.array(test_y))
        total_cost = len(test_y)

        #print(pooled_tweets_id)
        #print(len(pooled_tweets_id), len(test_y))

        #print(test_X[0], test_y[0])
        pooled_tweets_label = []
        for tweet_id in pooled_tweet_ids:
            pooled_tweets_label.append(test_y[tweet_id])

        current_hate = np.count_nonzero(np.array(pooled_tweets_label))
        recall_value = current_hate*1.0/total_hate

        budget_incur_ratio = pooled_cost*1.0/total_cost

        print(pool_depth, pooled_cost, total_cost, budget_incur_ratio, current_hate, total_hate, recall_value)

