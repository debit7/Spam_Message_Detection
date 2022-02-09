import pandas as pd
with open('textMsgs.data') as f:
        lst = []
        for ele in f:
            line = ele.replace('\n','').split('\t')
            
            lst.append(line)
Headers=['Classifier','Messages']
df = pd.DataFrame(lst,columns =Headers) 
print(df)