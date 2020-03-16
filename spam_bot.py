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
            testMsgBody.append(line[4:len(line)])
        elif line.find('ham') >= 0:
            testMsgType.append('H')
            testMsgBody.append(line[3:len(line)])

    else:
        if line.find('spam') >= 0:
            msgType.append('S')
            msgBody.append(line[4:len(line)])

        elif line.find('ham') >= 0:
            msgType.append('H')
            msgBody.append(line[3:len(line)])

    line = textFile.readline()
    count += 1


msgData = pd.DataFrame(list(zip(msgType, msgBody)), columns=['MsgType','MsgBody'])
testMsgData = pd.DataFrame(list(zip(testMsgType, testMsgBody)), columns=['MsgType','MsgBody'])
# print(msgData)
# print(testMsgData)

# Each test set is made up of a message type ( Ham OR Spam) & a message body (" mesgdsfsf sdf sd!.")
# We want to test each word in testData against each word of each message in messageData to see the probability of it being Ham OR Spam...

#  Algorithm:
# 
#   alpha = pseudo-count (we are using 0.2)
#   N = samples (we are using 20000)
# 
#                        count(word,ham/spam) + alpha
#  P(word|ham/spam) =   -------------------------------
#                        count(ham/spam) + N * alpha
# 
#  What is a word?
#  Avoid all special characters like : (?;@3$%!) except (I'll) the '

# def count( word, inType, inData):
#     count = 0

#     # for i in range(DataFrame):
    
#     print(inData)
#     return count

import string

N = 20000
alpha = 0.2

msg = testMsgData.loc[0][1]
print("msg:", msg)
words = msg.split()
print("words:", words)
print('word:', words[0])
print(string.punctuation)

print(len(testMsgData.loc[0][1].split()))




# for each in range(len(testMsgData)):
#     print(testMsgData.loc[each][1])

#     count(testData,"ham",msgData)

