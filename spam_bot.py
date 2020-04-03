import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer

message = pd.read_csv('./data/SMSSpamCollection',sep='\t',names=["labels","message"])

test_data = []
training_data = []

for i in range(len(message)):
    if(i % 10 == 0):
        test_data.append(message.iloc[i])
    else:
        training_data.append(message.iloc[i])

pd_test_data = pd.DataFrame(test_data)
pd_training_data = pd.DataFrame(training_data)


# Remove Punctuation & Case for Training Set 
for i in range(len(pd_training_data)):
    mess = str(pd_training_data.iloc[i]['message'])
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    pd_training_data.iloc[i]['message'] = nopunc.lower()

pd_training_data.reset_index(inplace = True)

# Remove Punctuation for Testing Set
for i in range(len(pd_test_data)):
    mess = str(pd_test_data.iloc[i]['message'])
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    pd_test_data.iloc[i]['message'] = nopunc.lower()

pd_test_data.reset_index(inplace = True)

# Gather Vocab for Messages
vocab_dictionary = {}
for index in range(len(pd_training_data)):
    for word in pd_training_data.iloc[index]["message"].split():
        if not (word in vocab_dictionary):
            vocab_dictionary[index] = word


from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
count_vector.fit(pd_training_data['message'])
count_vector.get_feature_names()

doc_array = count_vector.transform(pd_training_data['message']).toarray()

frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
frequency_matrix = frequency_matrix.join(pd_training_data['labels'])

# Group Ham and Spam
groups = frequency_matrix.groupby('labels')

ham_features = groups.get_group("ham")
ham_features.reset_index(drop = True, inplace = True)
ham_features = ham_features.drop(columns = ['labels'], axis = 1)

spam_features = groups.get_group("spam")
spam_features.reset_index(drop = True, inplace = True)
spam_features = spam_features.drop(columns = ['labels'], axis = 1)


# key = word, value = probablity 
# Calculate P(Ham|Word)
ham_words = list(ham_features)
ham_prob_dict = {}

for label in ham_words:
    feature_Total = ham_features[label].sum()
    ham_prob_dict[label] = float(feature_Total)/ len(ham_features)

spam_words = list(spam_features)
spam_prob_dict = {}

# Calculate P(Spam|Word)
for label in spam_words:
    feature_Total = spam_features[label].sum()
    spam_prob_dict[label] = (feature_Total) / len(spam_features)


predictions = []
# Calculate the probability of Ham and Spam
prob_ham = len(ham_features) / len(message)
prob_spam = len(spam_features) / len(message)

print("test:",ham_prob_dict['register'])

for index in range(len(pd_test_data)):
    msgProbOfHam = 1
    msgProbOfSpam = 1
    for word in pd_test_data.iloc[index]['message'].split():
        if word in ham_prob_dict:
       
            msgProbOfHam *= ham_prob_dict[word]
        if word in spam_prob_dict:
            
            msgProbOfSpam *= spam_prob_dict[word]

    ham_predict = (prob_ham * msgProbOfHam) / ( (prob_ham * msgProbOfHam) + (prob_spam * msgProbOfSpam) ) 
    spam_predict = (prob_spam * msgProbOfSpam) / ( (prob_spam * msgProbOfSpam) + (prob_ham * msgProbOfHam) )

    if ham_predict > spam_predict:
        predictions.append('ham')
    else:
        predictions.append('spam')


y_true = pd_test_data["labels"].tolist()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("CONFUSION MATRIX:")
print(confusion_matrix(y_true, predictions))

print("Classification Report")
print(classification_report(y_true, predictions))

