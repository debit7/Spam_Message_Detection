import pandas as pd
import math as mt
with open('textMsgs.data') as f:
        lst = []
        for ele in f:
            line = ele.replace('\n','').split('\t')
            
            lst.append(line)
Headers=['Classifier','Messages']
df = pd.DataFrame(lst,columns =Headers) 

#randomly splitting data in training and testing data set
training = df.sample(frac=0.8)
test = df.drop(training.index)
#removing punctuations from the messages and converting the upper cases to lower case
punctuations='''€˜%^&"\,!*_~)(-[};:]{'<#£$>./?@'''
stop_words=["i","da","we","ur" ,"u","am","me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
wordlist=[]
for message in training['Messages']:
    
    #print(message,'\n')
    for alpha in message:
        if alpha in punctuations:
            message = message.replace(alpha, " ")
    #for lower case        
    message=message.lower()
    
    #for stop words
    message=message.split()
    for word in message:
        if word in stop_words:
            #print(word,'\n')
            message.remove(word)
        wordlist.append(word)
wordlist = list(set(wordlist))

word_counts_per_Messages = pd.DataFrame([[row[1].count(word) for word in wordlist]
for _, row in training.iterrows()], columns=wordlist)
training = pd.concat([training.reset_index(), word_counts_per_Messages], axis=1).iloc[:,1:]
training.head()

# word_counts_per_Messages = [[row[1].count(word) for word in wordlist] for _, row in training.i
# print(word_counts_per_Messages)
# for index, sms in enumerate(training['Messages']):
#    for word in sms:
#       word_counts_per_Messages[word][index] += 1
# # training = pd.concat([training.reset_index(), word_counts_per_Messages], axis=1).iloc[:,1:]
# word_counts = pd.DataFrame(word_counts_per_Messages)
# word_counts.head()
print(word_counts_per_Messages)

Nvoc = len(training.columns) - 3
Pspam = training['Classifier'].value_counts()['spam'] / training.shape[0]
Pham = training['Classifier'].value_counts()['ham'] / training.shape[0]
Nspam = training.loc[training['Classifier'] == 'spam', 'Messages'].apply(len).sum()
Nham = training.loc[training['Classifier'] == 'ham', 'Messages'].apply(len).sum()


def p_w_spam(word):
    if word in training.columns:
        return (training.loc[training['Classifier'] == 'spam', word].sum() + 1) / (Nspam + Nvoc)
    else:
        return 1
def p_w_ham(word):
    if word in training.columns:
        return (training.loc[training['Classifier'] == 'ham', word].sum() + 1) / (Nham + Nvoc)
    else:
        return 1
def classify(message):
    p_spam_given_message = Pspam
    p_ham_given_message = Pham
    for word in message:
        p_spam_given_message *= p_w_spam(word)
        p_ham_given_message *= p_w_ham(word)
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'needs human classification'
def classify_improvization(message):
    p_spam_given_message = Pspam
    p_ham_given_message = Pham
    for word in message:
        p_spam_given_message += mt.log(p_w_spam(word),10)
        p_ham_given_message += mt.log(p_w_ham(word),10)
    p_spam_given_message=10**p_spam_given_message
    p_ham_given_message=10**p_ham_given_message
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'needs human classification'
test['predicted'] = test['Messages'].apply(classify_improvization)
print(test.head())

correct = (test['predicted'] == test['Classifier']).sum() / test.shape[0] * 100

print(correct)





#wordlist
   # print(message,'\n')
   # print(len(message),'\n')