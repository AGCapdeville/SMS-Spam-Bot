import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


textFile = open('./data/SMSSpamCollection','r')

msgType = []
msgBody = []

testMsgType = []
testMsgBody = []

line = textFile.readline()
count = 0

neighbors = 5
knn = KNeighborsClassifier(n_neighbors=neighbors)

while line:
    if count % 10 == 0:
        if line.find('spam') >= 0:
            testMsgType.append('S')
            testMsgBody.append(line[5:len(line)])
        elif line.find('ham') >= 0:
            testMsgType.append('H')
            testMsgBody.append(line[4:len(line)])

    else:
        if line.find('spam') >= 0:
            msgType.append('S')
            msgBody.append(line[5:len(line)])

        elif line.find('ham') >= 0:
            msgType.append('H')
            msgBody.append(line[4:len(line)])

    line = textFile.readline()
    count += 1


msgData = pd.DataFrame(list(zip(msgType, msgBody)), columns=['MsgType','MsgBody'])
testMsgData = pd.DataFrame(list(zip(testMsgType, testMsgBody)), columns=['MsgType','MsgBody'])


# print(msgData)
# print(testMsgData)

# Each test set is made up of a message type ( Ham OR Spam) & a message body (" mesgdsfsf sdf sd!.")
# We want to test each word in testData against each word of each message in messageData to see the probability of it being Ham OR Spam...

# TODO:
#  Algorithm: HAM OR SPAM
# 
#   alpha = pseudo-count (we are using 0.2)
#   N = samples (we are using 20000)
# 
#                        count(word,ham) + alpha
#  P(word|ham) =   -------------------------------
#                        count(ham) + N * alpha
# 
#             - - - - < OR > - - - -
# 
#                        count(word,spam) + alpha
#  P(word|spam) =   -------------------------------
#                        count(spam) + N * alpha
# 
# 
# https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier

# 
#  What is a word? Bird is a word. 
#  Avoid all special characters like : (?;@3$%!) except (I'll) the '



# def count( word, inType, inData):
#     count = 0
#     # for i in range(DataFrame):    
#     print(inData)
#     return count

import string

N = 20000
alpha = 0.2

# msg = testMsgData.loc[0][1]
# print("msg:", msg)
# words = msg.split()
# print("words:", words)
# print('word:', words[0])
# print(string.punctuation)

# REMOVING PUNCTUATION:
for i in range(len(msgData)):
    mess = str(msgData.iloc[i]['MsgBody'])
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    msgData.iloc[i]['MsgBody'] = nopunc

for i in range(len(testMsgData)):
    mess = str(msgData.iloc[i]['MsgBody'])
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    testMsgData.iloc[i]['MsgBody'] = nopunc


#  Probability of SPAM/HAM
# 
#                 COUNT(HAM MESSAGES) + K
#   P(HAM) =   ----------------------------
#                 N(TOTAL) + 1*CLASSES

#  CLASSES = 2, 


# Msg DATA HAM & SPAM COUNTS
groups = msgData.groupby('MsgType')
hamCount = groups.get_group("H").count().values[0]
spamCount = groups.get_group("S").count().values[0]





# print( len(msgData) )
# print( msgData.iloc[len(msgData)-1][1] )

# print(len(testMsgData))
