#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:32:07 2019
@author: jack
"""
import pandas as pd
import  numpy as np
import re
import matplotlib.pyplot as plt
from statistics import mean
#import matplotlib
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import ensemble
from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# 'spirit prior', 'spirit after', gender
features = [ 'type','rel prior', 'rel lit', 'rel service',  'rel prior', 'rel service', \
            'strength rel conv', 'rel influ','mar stat prior', 'mar stat after', \
            'relship change', 'conv strength', 'content of NDE','agegroup', \
            'changed', 'after dark']
#features = [ 'content of NDE']

drop_list = ['Unnamed: 0', 'case', 'race', 'heritage', 'c’ship', 'comm sz', \
             'comments prior', 'comm after', 'income', 'educ']

colors = ['r','y','g','b','c','k','m']
nextQ = 'Q.'

    
def show_results(total, wrong, model):
    """ provide a formatted comment output for the model accuracy data
    INPUT:  total: total data count
            wrong: count of incorrect predictions
            model:  label of the processing model
    OUTPUT: print statement
    """
    print ('{:14s}: {:5d} / {:5d} {:7.3f}%'.format(model, total-wrong,   \
           total, round(100*(total-wrong)/total,4)))
    
def graph_feature_importance(clf,df,label,y_col): 
    """ produce the graph of feature importance
        INPUT: gradient booster classifer & dataframe to get Y axis labels
        OUTPUT: plot of feature importance
    """
    feature_importance = clf.feature_importances_[0:9] #use the top 7
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
#    print("sorted index of top values:",sorted_idx)
#    print (df.columns[sorted_idx])
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, df.columns[sorted_idx])
    plt.xlabel(label)
    title = "Variable Importance\n" + y_col
    plt.title(title)
    plt.show()
    
def try_models(df,features,label,y_col):
    y=df[y_col]
    z = df.loc[:, features]
    #z.drop(columns=y_col,inplace=True)
    print(len(z.columns.tolist()),"columns")
    x = pd.get_dummies(z,columns=features)
    print("len dummies col list",len(x.columns.tolist()))
    #y=df['after dark']
    #y=df[y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.9, random_state=0) 
    print("data set sizes:",len(X_train),len(X_test),len(Y_train),len(Y_test))
    
    model="Random Forest"
    #print(model)
    rfc = ensemble.RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    y_pred = rfc.predict(X_test)# Classify, storing the result in a new variable.
    show_results(X_test.shape[0], (Y_test!= y_pred).sum(), model)
    
    model='multinomial'
    #print(model)
    bnb = MultinomialNB()
    bnb.fit(X_train,Y_train)
    y_pred = bnb.predict(X_test)# Classify, storing the result in a new variable.
    show_results(X_test.shape[0], (Y_test!= y_pred).sum(), model)
    
    model='gaussianNB '
    #print(model)
    bnb = GaussianNB()
    bnb.fit(X_train,Y_train)
    y_pred = bnb.predict(X_test)# Classify, storing the result in a new variable.
    show_results(X_test.shape[0], (Y_test != y_pred).sum(), model)

    model='BernoulliNB'
    #print(model)
    bnb = BernoulliNB()
    bnb.fit(X_train,Y_train)
    y_pred = bnb.predict(X_test)# Classify, storing the result in a new variable.
    show_results(X_test.shape[0], (Y_test != y_pred).sum(), model)

    model="Gradient   "
    #print(model)
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)# Classify, storing the result in a new variable.
    show_results(X_test.shape[0], (Y_test != y_pred).sum(), model)
    
    model="Random Forest"
    rfc = ensemble.RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    y_pred = rfc.predict(X_test)# Classify, storing the result in a new variable.
    show_results(X_test.shape[0], (Y_test!= y_pred).sum(), model)
    
    graph_feature_importance(rfc, x,label,y_col) # Graph importances relative to max importance.
    
    
  #START HERE  
fn="data"
df = pd.read_csv(fn)
df = df.drop(columns=drop_list)

#GENDER
fm = df['gender'].value_counts().to_dict()
labels = 'Male','Female'
sizes = [fm["M"] , fm["F"]]
hues= ['gold', 'yellowgreen']
plt.subplot(1, 2, 1)
plt.title("Gender")
plt.pie(sizes, labels=labels, colors=hues,\
autopct='%1.1f%%', shadow=True, startangle=140)

labels=[]
sizes=[]
fm = (df['agegroup'].value_counts().to_dict())
labels = 'Youth','Adult','Senior'
sizes = [fm["youth"] , fm["adult"], fm['senior']]#male,female]
hues= ['gold', 'yellowgreen','blue']
plt.subplot(1, 2, 2)
plt.title("Ages")
plt.pie(sizes, labels=labels, colors=hues,\
autopct='%1.1f%%', shadow=True, startangle=40)
plt.show() 

#PLOT relationship changes
i=0
x=[]
y=[]
total = len(df)

title = "Did Relationships Change?"
data = df['relship change'].value_counts().to_dict()
for ans, count in data.items():
    pct = round(100*count/total,3)
    if count > 1:
        y.append(pct)
        x.append(ans)
plt.subplot(1, 2, 1)
plt.title(title)
plt.bar(x,y,color=colors[i])

# MARRIAGE STATUS
title = "Marital Status %"
x=[]
y=[]
data = df['mar stat prior'].value_counts().to_dict()
for ans, count in data.items():
    pct = round(100*count/total,3)
    if count > 1:
        y.append(pct)
        x.append(ans)
#        print('{:7s} {:3d}  {:7.2f}%'.format(ans, count, pct))#pct))

ind = np.arange(len(x))    
width = 0.35

plt.subplot(1, 2, 2)
plt.title(title)
plt.bar(ind,y,width,color=colors[1],label="Before")
x=[]
y=[]
data = df['mar stat after'].value_counts().to_dict()
for ans, count in data.items():
    pct = round(100*count/total,3)
    if count > 1:
        y.append(pct)
        x.append(ans)
ind = np.arange(len(x)) 
plt.bar(ind+width,y,width,color=colors[5],label="After")
plt.xticks(ind + width / 2, x,rotation=90)
plt.legend(loc='best')
plt.show()

#SPIRITUALITY
title = "% Spiritual/Religious v. Non"
x=[]
y=[]
data = df['spirit prior'].value_counts().to_dict()
for ans, count in data.items():
    pct = round(100*count/total,3)
    if count > 1:
        y.append(pct)
        x.append(ans)
#        print('{:7s} {:3d}  {:7.2f}%'.format(ans, count, pct))#pct))
ind = np.arange(len(x) )    
width = 0.35
plt.subplot(1, 2, 1)
plt.title(title)
plt.bar(ind,y,width,color=colors[4],label="Before")


title = "Spirituality after the NDE"
x=[]
y=[]
data = df['spirit after'].value_counts().to_dict()
for ans, count in data.items():
    pct = round(100*count/total,3)
    if count > 1:
        y.append(pct)
        x.append(ans)
#        print('{:7s} {:3d}  {:7.2f}%'.format(ans, count, pct))#pct))
 
ind = np.arange(len(x))
plt.bar(ind+width,y,width,color=colors[5],label="After")
plt.subplot(1, 2, 1)
plt.title(title)
plt.xticks(ind + width / 2, x,rotation=90)
plt.legend(loc='best')
plt.show()
 
total = len(df)

df['pr light'] = df['pr dark'] = df['after light'] = df['after dark'] = 0
l_d=l_l=d_l=d_d=0
df['changed'] =0
 
for i, row in df.iterrows():
    df.at[i, 'pr light']  = int(('spiri' in row['spirit prior']) or \
         ('relig' in row['spirit prior']))
    df.at[i, 'pr dark'] =  int(('agnos' in row['spirit prior']) or \
         ('athe' in row['spirit prior']))
    
    df.at[i, 'after light']  = int(('spiri' in row['spirit after']) or \
         ('relig' in row['spirit after']))
    df.at[i, 'after dark'] =  int(('agnos' in row['spirit after']) or \
         ('athe' in row['spirit after']))

for i, row in df.iterrows():
    if row['pr light'] and row['after light']:
        l_l=l_l+1
    if row['pr light'] and row['after dark']:
        l_d=l_d+1
        df.at[i,'changed'] = 1
        
    if row['pr dark'] and row['after light']:
        d_l=d_l+1
        df.at[i,'changed'] = 1
    if row['pr dark'] and row['after dark']:
        d_d=d_d+1
  

title = "Perspective Changed"
light_after = (100*(l_l+d_l)/total)
dark_after = (100*(l_d+d_d)/total)
labels = 'Light to Dark', 'Light to Light','Dark to Light', 'Dark to Dark'
sizes = [l_d,l_l,d_l,d_d]
colors = ['lightcoral', 'lightskyblue', 'gold','green']
plt.subplot(1, 2, 1)
plt.title(title)
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
#plt.show()      

spaces = "                      "
title = spaces + "Found God, or Not, after the NDE"
light_after = (100*(l_l+d_l)/total)
dark_after = (100*(l_d+d_d)/total)
labels = 'Found After', 'Not Found After'
sizes = [light_after, dark_after]
colors = ['lightcoral', 'lightskyblue']
plt.subplot(1, 2, 2)
plt.title(title)
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=40)
plt.show()      
        
#print("\nLight after SDE:   {:7.2f}%".format(100*(l_l+d_l)/total))
#print("Dark  after SDE:   {:7.2f}%".format(100*(l_d+d_d)/total))

#STRENGTH OF CONVICTIONS
title = "Spirituality after the NDE"
spirit_chan = spirit_unch = 0
for index, row in df.iterrows():
    if row['spirit prior'] == row['spirit after']:
        spirit_unch = spirit_unch +1
    else:
        spirit_chan = spirit_chan+1

un = spirit_unch/(spirit_chan + spirit_unch)*100
chan = spirit_chan/(spirit_chan + spirit_unch)*100
labels = 'Changed', 'Unchanged'
sizes = [chan,un]
colors = ['gold', 'yellowgreen']
plt.subplot(1, 2, 1)
plt.title(title)
plt.pie(sizes, labels=labels, colors=colors,\
autopct='%1.1f%%', shadow=True, startangle=0)
#print('\n\t\tSpirituality \nchanged:{:7.2f}%, unchanged: {:7.2f}%'.format(un,chan))

#STRENGTH OF CONVICTIONS
title = spaces + "Strength of Religious Convictions"
labels = []
data = df['conv strength'].value_counts().to_dict()
for i, j in data.items():
    labels.append(i[:i.find(nextQ)])
xx =  dict((k, v) for k, v in data.items() if v > 2)
x=list(xx.values())
labels = labels[:len(x)]
plt.yticks(fontsize=6)
col_lst = ['gold', 'yellowgreen','lightcoral', 'lightskyblue','green']
plt.subplot(1, 2, 2)
plt.title(title)
plt.pie(x, labels=labels, colors=col_lst,autopct='%1.1f%%', shadow=True, startangle=0)
plt.show()

#CONTENT
title = "Content of Experience"
labels = []
data = df['content of NDE'].value_counts().to_dict()
print(data)
for i, j in data.items():
    labels.append(i[:i.find(nextQ)])

x=list(data.values())
col_list  = colors[:len(y)]
plt.subplot(1, 2, 2)
plt.title(title)
plt.pie(x, labels=labels, colors=col_lst,autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()

features = [ 'type','rel prior', 'rel lit', 'rel service',  'rel prior', 'rel service', \
            'strength rel conv', 'rel influ','mar stat prior', 'mar stat after', \
            'relship change', 'conv strength', 'content of NDE','agegroup']
try_models(df,features,"Importance of all Factors", 'after dark')

features = [ 'rel prior', 'rel lit', 'rel service',  'rel prior', 'rel service', \
            'strength rel conv', 'rel influ','mar stat prior', 'mar stat after', \
            'relship change', 'conv strength', 'content of NDE','agegroup']
try_models(df,features,"With Type Removed",'after dark')
print("types:\n",df['type'].value_counts())#.to_dict())



#Process dark words
dark_words = (open('dark words.txt', 'r').read())
df['dark words'] = 0
y=n=c=0
for index, each in df.iterrows():
        c=c+1
        for w in dark_words.split():
            if each['detail'].find(re.sub("\'","", w)) > 0:
                df.at[index,'dark words'] = df.at[index,'dark words'] +1
                y=y+1
            else:
                n=n+1
print ("c:",c,"y:",y,"n:",n)

fn = "output data"
df.to_csv(fn)

    
title = "Dark Words V. Dark State"
#y =  minmax_scale(df[df['after light']==1]['dark words'])
#x =  minmax_scale(df[df['after dark']==1]['dark words'])

l =  (df[df['after light']==1]['dark words'])
d =  (df[df['after dark']==1]['dark words'])
print ("l:",mean(l))
print ("d:",mean(d))


plt.hist(l,bins=20,color='b',alpha=0.3,label='light',histtype='stepfilled', normed=True)
plt.hist(d,bins=20,alpha=0.5,color='g',label='dark',histtype='stepfilled',normed=True)
plt.xlabel("Dark Word Occurrences")
plt.ylabel("Normalised Frequency")
plt.legend()
plt.show()

plt.plot(l,label="light")
plt.show
plt.plot(d,label="dark")
plt.legend()
plt.show
#plt.scatter(x, y)#, s=area, c=colors, alpha=0.5)
#plt.plot(x,  'b--')
#plt.plot(y, 'g-.')
#plt.subplot(1, 2, 1)
#plt.bar(x,y,color=colors[2])
"""
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('y1')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('y2', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.bar(x,y,color=colors[1])
"""
