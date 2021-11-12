import numpy as np
import pandas as pd
from sklearn import preprocessing, mixture, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix, f1_score

# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


def getEncode(df, name, encoder):
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels


# onehot Encoding
def onehotEncode(df, name):
    le = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc = df[[name]]
    enc = le.fit_transform(enc).toarray()
    le.categories_[0] = le.categories_[0].astype(np.str)
    new = np.full((len(le.categories_[0]), 1), name + ": ")
    le.categories_[0] = np.core.defchararray.add(new, le.categories_[0])
    enc_df = pd.DataFrame(enc, columns=le.categories_[0][0])
    df.reset_index(drop=True,inplace=True)
    df = pd.concat([df, enc_df], axis=1)
    df.drop(columns=[name], inplace=True)
    return df


# label encoding
def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels




"""
Function to get 2d array of dataframe with given dataframe
output: scale/encoded 2d array dataframe, scalers used, encoders used
"""
def get_various_encode_scale(X, numerical_columns, categorical_columns, scalers=None, encoders=None, scaler_name=None,
                             encoder_name=None):
    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []

    if len(categorical_columns) == 0:
        return get_various_scale(X, numerical_columns, scalers, scaler_name)
    if len(numerical_columns) == 0:
        return get_various_encode(X, categorical_columns, encoders, encoder_name)

    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(), preprocessing.OneHotEncoder()]

    after_scale_encode = [[0 for col in range(len(encoders))] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale_encode[i].pop()
        for encode in encoders:
            after_scale_encode[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                after_scale_encode[i][j][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
                j = j + 1
            i = i + 1

    for new in categorical_columns:
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                    labelEncode(after_scale_encode[i][j], new)
                elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                    after_scale_encode[i][j] = onehotEncode(after_scale_encode[i][j], new)
                else:
                    getEncode(after_scale_encode[i][j], new, encoder)
                j = j + 1
            i = i + 1

    return after_scale_encode, scalers, encoders


"""
If there aren't categorical value, do this function
This function only scales given X
Output: 1d array of scaled dataset, scalers used, encoders used(Nothing)
"""
def get_various_scale(X, numerical_columns, scalers=None, scaler_name=None):
    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
        # scalers = [preprocessing.StandardScaler()]
    encoders = ["None"]

    after_scale = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale[i].pop()
        for encode in encoders:
            after_scale[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
        i = 0
        for scaler in scalers:
            after_scale[i][0][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
            i = i + 1

    return after_scale, scalers, ["None"]


"""
If there aren't numerical value, do this function
This function only encodes given X
Return: 1d array of encoded dataset, scalers used(Nothing), encoders used
"""
def get_various_encode(X, categorical_columns, encoders=None, encoder_name=None):
    """
    Test scale/encoder sets
    """
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]
        # encoders = [preprocessing.LabelEncoder()]
    scalers = ["None"]

    after_encode = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_encode[i].pop()
        for encode in encoders:
            after_encode[i].append(X.copy())
        i = i + 1

    for new in categorical_columns:
        j = 0
        for encoder in encoders:
            if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                labelEncode(after_encode[0][j], new)
            elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                after_encode[0][j] = onehotEncode(after_encode[0][j], new)
            else:
                getEncode(after_encode[0][j], new, encoder)
            j = j + 1

    return after_encode, ["None"], encoders

"""
Function that starts automated machine learning
Compare various parameters(classifiers,scalers, encoders, columns, classifier parameters)
Return score compared given test target with predicted target 
"""
def get_Result(X, y,test_x,test_y, numerical_columns, categorical_columns, n_split_time=None,n_jobs=None):

    model, params, indices, numerical, categorical = do_Classify(X, y, numerical_columns, categorical_columns, n_split_time,n_jobs)
    print("\n=====================================")
    print("\nHighest Score:")
    print("Model: ",model)
    for param,index in zip(params,indices):
        print(index," : ",param)
    print("Numerical columns: ",numerical)
    print("Categorical columns: ",categorical)

    new_x = pd.DataFrame()
    data = X.copy()
    new_test = pd.DataFrame()
    test_data = test_x.copy()
    Foldnum = 5

    for numerical_column_ind in numerical:
        new_x.loc[:, numerical_column_ind] = data.loc[:, numerical_column_ind]
    for categorical_column_ind in categorical:
        new_x.loc[:, categorical_column_ind] = data.loc[:, categorical_column_ind]

    for numerical_column_ind in numerical:
        new_test.loc[:, numerical_column_ind] = test_data.loc[:, numerical_column_ind]
    for categorical_column_ind in categorical:
        new_test.loc[:, categorical_column_ind] = test_data.loc[:, categorical_column_ind]

    encoder = [params.pop()]
    scaler = [params.pop()]


    new_x, scalers, encoders = get_various_encode_scale(new_x, numerical, categorical,scalers=scaler,encoders=encoder)
    new_test, scalers, encoders = get_various_encode_scale(new_test, numerical, categorical, scalers=scaler, encoders=encoder)

    model_param = {}
    for param,index in zip(params,indices):
        model_param[index] = param
    model.set_params(**model_param)
    model.fit(new_x[0][0],y)

    for col in new_x[0][0].columns.difference(new_test[0][0].columns):
        new_test[0][0].loc[:,col] = 0

    y_pred = model.predict(new_test[0][0])

    compare_y = pd.DataFrame({'y_pred': y_pred, 'y_test': test_y})

    print("\nCompare Prediction and Test")
    print(compare_y)
    return ((y_pred == test_y).sum() / len(compare_y))

"""
Function that do classify with given several parameters
Return parameters that got most highest score.
"""
def do_Classify(X, y, numerical_columns, categorical_columns, n_split_time=None,n_jobs=None):

    if n_jobs == None:
        n_jobs = -1
    if n_split_time == None:
        n_split_time = 5

    DT_best_score = 0
    DT_best_param = [0, 0, 'scaler', 'encoder']
    DT_best_index = ['max_depth', 'min_samples_split', 'scaler', 'encoder']
    DT_best_numerical_cols = []
    DT_best_categorical_cols = []

    LR_best_score = 0
    LR_best_param = [0, 0, 0, 'scaler', 'encoder']
    LR_best_index = ['max_iter', 'C','tol', 'scaler', 'encoder']
    LR_best_numerical_cols = []
    LR_best_categorical_cols = []

    RF_best_score = 0
    RF_best_param = [0, 0, 0, 'scaler', 'encoder']
    RF_best_index = ['n_estimators', 'max_features', 'max_depth', 'scaler', 'encoder']
    RF_best_numerical_cols = []
    RF_best_categorical_cols = []

    for numerical_column, categorical_column in zip(numerical_columns, categorical_columns):

        new_x = pd.DataFrame()
        data = X.copy()
        Foldnum = 5

        for numerical_column_ind in numerical_column:
            new_x.loc[:, numerical_column_ind] = data.loc[:, numerical_column_ind]
        for categorical_column_ind in categorical_column:
            new_x.loc[:, categorical_column_ind] = data.loc[:, categorical_column_ind]

        new_x, scalers, encoders = get_various_encode_scale(new_x, numerical_column, categorical_column)

        i=0
        for scaler in scalers:
           j=0
           for encoder in encoders:

              x_train, x_test, y_train, y_test = train_test_split(new_x[i][j], y, test_size=1 / Foldnum, shuffle=True)

              max_depth,min_sample_split,get_f1_score = get_Best_Decision_Tree_Classifier(x_train,x_test,y_train,y_test)
              if(get_f1_score>DT_best_score):
                  DT_best_param[0] = max_depth
                  DT_best_param[1] = min_sample_split
                  DT_best_param[2] = scaler
                  DT_best_param[3] = encoder
                  DT_best_score = get_f1_score
                  DT_best_numerical_cols = numerical_column
                  DT_best_categorical_cols = categorical_column

              max_iter, C, tol,get_f1_score = get_Best_Logistic_Regression(x_train, x_test, y_train, y_test)
              if (get_f1_score > LR_best_score):
                  LR_best_param[0] = max_iter
                  LR_best_param[1] = C
                  LR_best_param[2] = tol
                  LR_best_param[3] = scaler
                  LR_best_param[4] = encoder
                  LR_best_score = get_f1_score
                  LR_best_numerical_cols = numerical_column
                  LR_best_categorical_cols = categorical_column

              n_estimator, max_features, max_depth, get_f1_score = get_Best_Random_Forest(x_train, x_test, y_train, y_test)
              if (get_f1_score > RF_best_score):
                  RF_best_param[0] = n_estimator
                  RF_best_param[1] = max_features
                  RF_best_param[2] = max_depth
                  RF_best_param[3] = scaler
                  RF_best_param[4] = encoder
                  RF_best_score = get_f1_score
                  RF_best_numerical_cols = numerical_column
                  RF_best_categorical_cols = categorical_column

              j=j+1
           i=i+1

    print("Best Score and parameters for each classifiers\n=========================================")
    print("Decision Tree")
    print("Score: ",DT_best_score)
    for index,param in zip(DT_best_index,DT_best_param):
        print(index," : ", param)
    print("Columns: ",DT_best_numerical_cols,DT_best_categorical_cols)

    print("\nLogistic Regression")
    print("Score: ", LR_best_score)
    for index, param in zip(LR_best_index, LR_best_param):
        print(index, " : ", param)
    print("Columns: ", LR_best_numerical_cols, LR_best_categorical_cols)

    print("\nRandom Forest")
    print("Score: ", RF_best_score)
    for index, param in zip(RF_best_index, RF_best_param):
        print(index, " : ", param)
    print("Columns: ", RF_best_numerical_cols, RF_best_categorical_cols)

    if(DT_best_score>=LR_best_score and DT_best_score>=RF_best_score):
        return DecisionTreeClassifier(), DT_best_param,DT_best_index ,DT_best_numerical_cols,DT_best_categorical_cols
    elif(LR_best_score>=DT_best_score and LR_best_score>=RF_best_score):
        return LogisticRegression(), LR_best_param, LR_best_index, LR_best_numerical_cols, LR_best_categorical_cols
    elif(RF_best_score>DT_best_score and RF_best_score>=LR_best_score):
        return RandomForestClassifier(),RF_best_param, RF_best_index, RF_best_numerical_cols, RF_best_categorical_cols

"""
Function that find classifier parameter of decision tree classifier that gives highest score
Do Parameter tuning with default parameters and parameter growers
If the parameter does not change by more than the minimum amount change, stop tuning
"""
def get_Best_Decision_Tree_Classifier(X_train,X_test,y_train,y_test):

    max_depth = 10
    max_depth_grower = 2
    min_samples_split = 10
    min_samples_split_grower = 2
    pre_score = 0.0
    max_score= 0.0
    direction = 1
    #minimum amount change
    MAC = 0.000001

    max_score = get_DT_score(X_train,X_test,y_train,y_test,max_depth,min_samples_split)
    left_score = get_DT_score(X_train,X_test,y_train,y_test,max_depth-max_depth_grower,min_samples_split)
    right_score = get_DT_score(X_train,X_test,y_train,y_test,max_depth+max_depth_grower,min_samples_split)

    if(max_score>left_score and max_score>right_score):
        direction = 0
    elif(left_score>right_score):
        direction = -1
    elif(right_score>left_score):
        direction = 1

    while(max_score>pre_score and ((max_score-pre_score)/max_score)>MAC and max_depth + direction * max_depth_grower>0):
        pre_score = max_score
        max_depth = max_depth + direction * max_depth_grower
        max_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth, min_samples_split)

    max_score = pre_score
    max_depth = max_depth - direction * max_depth_grower

    left_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth,
                              min_samples_split-min_samples_split_grower)
    right_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth,
                               min_samples_split+min_samples_split_grower)

    if (max_score > left_score and max_score > right_score):
        direction = 0
    elif (left_score > right_score):
        direction = -1
    elif (right_score > left_score):
        direction = 1

    pre_score = 0

    while (max_score > pre_score and ((max_score-pre_score)/max_score)>MAC and min_samples_split + direction * min_samples_split_grower>0):
        if(min_samples_split + direction * min_samples_split_grower ==0):
            min_samples_split = min_samples_split + direction * min_samples_split_grower
            break;
        pre_score = max_score
        min_samples_split = min_samples_split + direction * min_samples_split_grower
        max_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth, min_samples_split)

    max_score = pre_score
    min_samples_split = min_samples_split - direction * min_samples_split_grower

    return max_depth,min_samples_split,max_score

def get_DT_score(X_train,X_test,y_train,y_test,max_depth,min_samples_split):

    model = DecisionTreeClassifier(random_state=47, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

"""
Function that find classifier parameter of Logistic regression that gives highest score
Do Parameter tuning with default parameters and parameter growers
If the parameter does not change by more than the minimum amount change, stop tuning
"""
def get_Best_Logistic_Regression(X_train,X_test,y_train,y_test):

    penalty = 'l2'
    solver = 'liblinear'
    max_iter = 300
    max_iter_grower = 5
    C = 10
    C_grower = 2
    tol = 1e-4
    tol_grower = 0.5
    direction = 1
    #minimum amount change
    MAC = 0.00001
    max_score = 0.0
    pre_score = 0.0

    #Max_iter
    max_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter,C,tol)
    left_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter-max_iter_grower,C,tol)
    right_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter+max_iter_grower,C,tol)

    if(max_score>=left_score and max_score>=right_score):
        direction = 0
    elif(left_score>right_score):
        direction = -1
    elif(right_score>left_score):
        direction = 1

    while(max_score>pre_score and ((max_score-pre_score)/max_score)>MAC and max_iter + direction * max_iter_grower > 0):
        pre_score = max_score
        max_iter = max_iter + direction * max_iter_grower
        max_score = get_LR_score(X_train, X_test, y_train, y_test,penalty,solver, max_iter, C,tol)
    max_score = pre_score
    max_iter = max_iter - direction * max_iter_grower

    # C
    left_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C-C_grower,tol)
    right_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C+C_grower,tol)

    if (max_score >= left_score and max_score >= right_score):
        direction = 0
    elif (left_score > right_score):
        direction = -1
    elif (right_score > left_score):
        direction = 1

    pre_score = 0
    while (max_score > pre_score and ((max_score-pre_score)/max_score)>MAC and C + direction * C_grower>0):
        pre_score = max_score
        C = C + direction * C_grower
        max_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter,C,tol)

    max_score = pre_score
    C = C - direction * C_grower

    # tol
    left_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C,tol*tol_grower)
    right_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C,tol/tol_grower)

    if (max_score >= left_score and max_score >= right_score):
        direction = 0
    elif (right_score > left_score):
        tol_grower = 1/tol_grower

    while (max_score > pre_score and ((max_score - pre_score) / max_score) > MAC and tol * tol_grower>0):
        pre_score = max_score
        tol = tol * tol_grower
        max_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C,tol)
    max_score = pre_score
    tol = tol / max_iter_grower

    return max_iter,C,tol,max_score

def get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter,C,tol):
    model = LogisticRegression(penalty=penalty,solver=solver, max_iter=max_iter,C=C,tol=tol)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')



"""
Function that find classifier parameter of random forest classifier that gives highest score
Do Parameter tuning with default parameters and parameter growers
If the parameter does not change by more than the minimum amount change, stop tuning
"""
def get_Best_Random_Forest(X_train,X_test,y_train,y_test):

    criterion = 'gini'
    n_estimators = 50
    n_estimators_grower = 5
    max_features = int(len(X_train.columns)/2)
    max_features_grower = 1
    max_depth = 50
    max_depth_grower=5

    direction = 1
    #minimum amount change
    MAC = 0.0000001
    max_score = 0.0
    pre_score = 0.0

    #n_estimator
    max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)
    left_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators-n_estimators_grower,max_features,max_depth)
    right_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators+n_estimators_grower,max_features,max_depth)

    if(max_score>=left_score and max_score>=right_score):
        direction = 0
    elif(left_score>right_score):
        direction = -1
    elif(right_score>left_score):
        direction = 1

    while(max_score>pre_score and ((max_score-pre_score)/max_score)>MAC and  n_estimators + direction * n_estimators_grower>0):
        pre_score = max_score
        n_estimators = n_estimators + direction * n_estimators_grower
        max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)
    max_score = pre_score
    n_estimators = n_estimators - direction * n_estimators_grower

    # max_features
    if (max_features + direction * max_features_grower > len(X_train.columns) and max_features + direction * max_features_grower == 0):
        left_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features-max_features_grower,max_depth)
        right_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features+max_features_grower,max_depth)

        if (max_score >= left_score and max_score >= right_score):
            direction = 0
        elif (left_score > right_score):
            direction = -1
        elif (right_score > left_score):
            direction = 1

        pre_score = 0
        while (max_score > pre_score and ((max_score-pre_score)/max_score)>MAC):
            if (max_features + direction * max_features_grower>len(X_train.columns) or max_features + direction * max_features_grower==0):
                max_features = max_features + direction * max_features_grower
                break
            pre_score = max_score
            max_features = max_features + direction * max_features_grower
            max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)

        max_score = pre_score
        max_features = max_features - direction * max_features


    # max_depth
    left_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth-max_depth_grower)
    right_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth+max_depth_grower)

    if (max_score >= left_score and max_score >= right_score):
        direction = 0
    elif (left_score > right_score):
        direction = -1
    elif (right_score > left_score):
        direction = 1

    while (max_score > pre_score and ((max_score - pre_score) / max_score) > MAC and max_depth * max_depth_grower>0):
        pre_score = max_score
        max_depth = max_depth * max_depth_grower
        max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)
    max_score = pre_score
    max_depth = max_depth - direction * max_depth

    return n_estimators,max_features,max_depth,max_score

def get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

    

    
    


