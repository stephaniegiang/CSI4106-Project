
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.tree import DecisionTreeClassifier
import datetime
from sklearn.neural_network import MLPClassifier


# In[2]:


X = pd.read_csv('data.csv')
# X.head(10)
y = X.pop("SalePrice").values
X.pop('Id') #not needed

X2 = pd.read_csv('test.csv')
ids = X2.pop('Id') #not needed

def split(number_of_features=10, seed = 0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

#     print(X_train.shape)
#     print(y_test.shape)
    random.seed=seed

    randomFeatures = random.sample(list(X), number_of_features)
#     print(randomFeatures)

    # TO DO - Finish the remaining encoding process

    X_train = X_train[randomFeatures].copy()
    X_test = X_test[randomFeatures].copy()
    
    return randomFeatures, X_train, X_test, y_train, y_test

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def replace_Na(data):
    clist = data.columns[data.isna().any()].tolist()
    for col in clist:
        index = data[col].first_valid_index()
        replace = 0
        if index:
            first_valid_value = data[col].loc[index]
            if not is_number(str(first_valid_value)):
                replace = ""
#         print(col,":",data[col].loc[index], "with -->",replace,"<--")
        data[col].fillna(replace,inplace=True)

def encode(train, test, out=False):
    replace_Na(train)
    replace_Na(test)
    if out:
        train.to_csv("traindata.csv", index=False, header=True)
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    ohe.fit(train)

    X_train_encoded = ohe.transform(train)
    X_test_encoded = ohe.transform(test)
    
    return X_train_encoded, X_test_encoded

def train_model(clf, X_train, y_train, epochs=3):
    scores = []
    print("Starting training...")
    for i in range(1, epochs + 1):
        print("Epoch:" + str(i) + "/" + str(epochs) + " -- " + str(datetime.datetime.now()))
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        scores.append(score)
    print("Done training.")
    return scores

def get_clf():
    return DecisionTreeClassifier(random_state=1)

def run_best(best_features, epochs, filename):
    clf = get_clf()
    # features, train, test, y_train, y_test = split(number_of_feat, seed)
    train = X[best_features].copy()
    test = X2[best_features].copy()

    print("***encoding Test Data***")

    train_encoded, test_encoded = encode(train, test, True)

    clf_scores = train_model(clf, train_encoded, y, epochs)

    print("\nDone training, Scores:\n", clf_scores)

    print("\npredicting and saving to",filename)

    y_predicted = clf.predict(test_encoded)

    df = pd.DataFrame(columns=['Id', 'SalePrice'])

    for i in range(len(y_predicted)):
        df.loc[i] = [ids[i],y_predicted[i]]

    df.to_csv(filename, index=None, header=True)



def run_sim():
    scores = 0
    bfeatures = []
    seed = 0
    number_of_feat = 0
    epochs = 4
    for i in range(10):
            
        #initially default parameters
        clf = get_clf()
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(150, 150), random_state=1, max_iter=150,
        #               learning_rate_init=0.1, warm_start=False)
        num_feat = random.randint(1,40)
        
        features, train, test, y_train, y_test = split(num_feat,i)
        
        print(i,"\n\n ***Running features:",num_feat,features,"***\n")
        
        train_encoded, test_encoded = encode(train, test)
        

        clf_scores = train_model(clf, train_encoded, y_train, epochs)
        
        print("\nDone training, Scores:\n",clf_scores)

        y_predicted = clf.predict(train_encoded[0:10])
        print("\n",y_predicted)
        print(y_train[0:10])
        # clf.predict_proba(train_encoded[0:10])
        
#         print("\nTesting\n\n")

        test_score = clf.score(test_encoded, y_test)

        if test_score > scores:
            bfeatures = features
            scores = test_score
            seed = i
            number_of_feat = num_feat

        print("\ncurrent Score:",test_score)
        print("\nbest Score:",scores,"\n")

    
    print("\n\n best features:", bfeatures)
    print('best score:',scores)
    print('seed:',seed)

    run_best(bfeatures,epochs,"predictions.csv")





# In[8]:


print("running")
run_sim()

