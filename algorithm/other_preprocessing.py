"""
In this python file multiple classification models are trained.
"""
import math
import copy
import time
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading, LabelPropagation
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy import stats
from .Fair_SMOTE.SMOTE import smote
from .Fair_SMOTE.Generate_Samples import generate_samples
from .evaluation.eval_classifier import acc_bias


class Preprocessing():
    """Multiple different model learners are part of this class.

    Parameters
    ----------
    X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
        Training data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_train: array-like, shape (n_samples)
        Label vector relative to the training data X_train.

    X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
        Test data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_test: array-like, shape (n_samples)
        Label vector relative to the test data X_test.

    sens_attrs: list of strings
        List of the column names of sensitive attributes in the dataset.
    """
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, dataset_train,
        dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
        index, label, metric, link, input_file, remove):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.testsize = testsize
        self.randomstate = randomstate
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.index = index
        self.label = label
        self.metric = metric
        self.X_tr_np = self.X_train.to_numpy()
        self.y_tr_np = self.y_train.to_numpy().reshape((self.y_train.to_numpy().shape[0],))
        self.X_te_np = self.X_test.to_numpy()
        self.y_te_np = self.y_test.to_numpy().reshape((self.y_test.to_numpy().shape[0],))
        self.link = link
        self.input_file = input_file
        self.remove = remove


    def smote(self, classifier, do_eval=False):
        label = list(self.y_train.columns)[0]
        train_df = copy.deepcopy(self.X_train)
        train_df[label] = copy.deepcopy(self.y_train)
        dataset_orig_train = copy.deepcopy(train_df)
        train_df.reset_index(drop=True, inplace=True)
        cols = train_df.columns
        smt = smote(train_df)
        train_df = smt.run()
        train_df.columns = cols
        y_train_new = train_df[label]
        X_train_new = train_df.drop(label, axis=1)

        dict_cols = dict()
        cols = list(train_df.columns)
        for i, col in enumerate(cols):
            dict_cols[i] = col
        #Find Class & protected attribute distribution
        zero_zero = len(dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 1)])
        maximum = max(zero_zero,zero_one,one_zero,one_one)
        if maximum == zero_zero:
            zero_one_to_be_incresed = maximum - zero_one
            one_zero_to_be_incresed = maximum - one_zero
            one_one_to_be_incresed = maximum - one_one
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_one)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_one_one)
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_new = df_new.append(df_zero_zero)
        if maximum == zero_one:
            zero_zero_to_be_incresed = maximum - zero_zero
            one_zero_to_be_incresed = maximum - one_zero
            one_one_to_be_incresed = maximum - one_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_one_one)
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_new = df_new.append(df_zero_one)
        if maximum == one_zero:
            zero_zero_to_be_incresed = maximum - zero_zero
            zero_one_to_be_incresed = maximum - zero_one
            one_one_to_be_incresed = maximum - one_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_zero_one)
            df_new = df_new.append(df_one_one)
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_new = df_new.append(df_one_zero)
        if maximum == one_one:
            zero_zero_to_be_incresed = maximum - zero_zero
            one_zero_to_be_incresed = maximum - one_zero
            zero_one_to_be_incresed = maximum - zero_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_zero_one)
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1) & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_new = df_new.append(df_one_one)

        X_train_new, y_train_new = df_new.loc[:, df_new.columns != label], df_new[label]
        X_test_new = copy.deepcopy(self.X_test)
        if self.remove:
            for sens in self.sens_attrs:
                X_train_new = X_train_new.drop(sens, axis=1)
                X_test_new = X_test_new.drop(sens, axis=1)
        classifier.fit(X_train_new, y_train_new)
        pred = classifier.predict(X_test_new)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train_new, y_train_new, metric_val, "Fair-SMOTE"


    def fairssl(self, classifier, ssl_type="SelfTraining", balancing=False, do_eval=False):
        label = list(self.y_train.columns)[0]
        train_df = copy.deepcopy(self.X_train)
        train_df[label] = copy.deepcopy(self.y_train)
        dataset_orig = copy.deepcopy(train_df)
        column_to_move = dataset_orig.pop(label)
        dataset_orig[label] = column_to_move

        dataset_orig_zero, dataset_orig_one = [x for _, x in dataset_orig.groupby(dataset_orig[self.sens_attrs[0]] == 0)]
        dataset_orig_zero[self.sens_attrs[0]] = 0
        X_train_zero, y_train_zero = dataset_orig_zero.loc[:, dataset_orig_zero.columns != label], dataset_orig_zero[label]
        clf_zero = copy.deepcopy(classifier)
        clf_zero.fit(X_train_zero, y_train_zero)
        X_train_one, y_train_one = dataset_orig_one.loc[:, dataset_orig_one.columns != label], dataset_orig_one[label]
        clf_one = copy.deepcopy(classifier)
        clf_one.fit(X_train_one, y_train_one)
        
        unlabeled_df = pd.DataFrame(columns=dataset_orig.columns)

        for index, row in dataset_orig.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_zero = clf_zero.predict(row_)
            y_one = clf_one.predict(row_)
            if y_zero[0] != y_one[0]:
                unlabeled_df = unlabeled_df.append(row, ignore_index=True)
                dataset_orig = dataset_orig.drop(index)

        dict_cols = dict()
        cols = list(train_df.columns)
        for i, col in enumerate(cols):
            dict_cols[i] = col

        zero_zero_df = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 0)]
        zero_one_df = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 1)]
        one_zero_df = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 0)]
        one_one_df = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 1)]

        x = int(len(self.X_train)/25)

        df1 = zero_zero_df[:x].append(zero_one_df[:x])
        df2 = df1.append(one_zero_df[:x])
        df3 = df2.append(one_one_df[:x])
        labeled_df = df3.reset_index(drop=True)
        labeled_df = shuffle(labeled_df)

        df1 = zero_zero_df[x:].append(zero_one_df[x:])
        df2 = df1.append(one_zero_df[x:])
        df3 = df2.append(one_one_df[x:])
        unlabeled_df = unlabeled_df.append(df3).reset_index(drop=True)

        unlabeled_df[label] = -1
        mixed_df = labeled_df.append(unlabeled_df)
        X_train, y_train = mixed_df.loc[:, mixed_df.columns != label], mixed_df[label]

        if ssl_type in ("SelfTraining", "LabelSpreading", "LabelPropagation"):
            if ssl_type == "SelfTraining":
                training_model = SelfTrainingClassifier(classifier)
                training_model.fit(X_train, y_train)
                model_name = "FairSSL-ST"
            elif ssl_type == "LabelSpreading":
                training_model = LabelSpreading()
                training_model.fit(X_train, y_train)
                model_name = "FairSSL-LS"
            elif ssl_type == "LabelPropagation":
                training_model = LabelPropagation()
                training_model.fit(X_train, y_train)
                model_name = "FairSSL-LP"
            X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != label], unlabeled_df[label]
            y_pred_proba = training_model.predict_proba(X_unl)
            y_pred = training_model.predict(X_unl)

            to_keep = []

            for i in range(len(y_pred_proba)):
                if max(y_pred_proba[i]) >= 0.6:
                    to_keep.append(i)

            X_unl_certain = X_unl.iloc[to_keep,:]
            y_unl_certain = y_pred[to_keep]

            X_train, y_train = labeled_df.loc[:, labeled_df.columns != label], labeled_df[label]

            X_train = X_train.append(X_unl_certain)
            y_train = np.concatenate([y_train, y_unl_certain])
        elif ssl_type == "CoTraining":
            X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != label], unlabeled_df[label]
            X_train, y_train = labeled_df.loc[:, labeled_df.columns != label], labeled_df[label]
            classifier_dict = {}
            for i in X_train.columns:
                classifier_dict[i] = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100).fit(X_train[[i]],y_train)
            length_of_columns = len(X_unl.columns)
            change = 0
            for index, row in X_unl.iterrows():
                try:
                    prediction_for_current_point = 0
                    for j in classifier_dict:
                        prediction_for_current_point += classifier_dict[j].predict([[row[j]]])[0]
                    if prediction_for_current_point > (length_of_columns//2) + 1:
                        y_unl.iloc[index] = 1
                        change += 1
                    elif prediction_for_current_point < (length_of_columns//2) - 1:
                        y_unl.iloc[index] = 0
                        change += 1
                    else:
                        X_unl = X_unl.drop(index)
                        y_unl = y_unl.drop(index)
                except:
                    break
            X_train = X_train.append(X_unl)
            y_train = y_train.append(y_unl)
            model_name = "FairSSL-CT"


        if balancing:
            X_train[label] = y_train
            dataset_orig = X_train

            zero_zero = len(dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 0)])
            zero_one = len(dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 1)])
            one_zero = len(dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 0)])
            one_one = len(dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 1)])
            maximum = max(zero_zero,zero_one,one_zero,one_one)

            if maximum == zero_zero:
                zero_one_to_be_incresed = maximum - zero_one
                one_zero_to_be_incresed = maximum - one_zero
                one_one_to_be_incresed = maximum - one_one
                df_zero_one = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_one_zero = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_one_one = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
                df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
                df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
                df_new = copy.deepcopy(df_zero_one)
                df_new = df_new.append(df_one_zero)
                df_new = df_new.append(df_one_one)
                df_zero_zero = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_new = df_new.append(df_zero_zero)
            if maximum == zero_one:
                zero_zero_to_be_incresed = maximum - zero_zero
                one_zero_to_be_incresed = maximum - one_zero
                one_one_to_be_incresed = maximum - one_one
                df_zero_zero = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_one_zero = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_one_one = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
                df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
                df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
                df_new = copy.deepcopy(df_zero_zero)
                df_new = df_new.append(df_one_zero)
                df_new = df_new.append(df_one_one)
                df_zero_one = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_new = df_new.append(df_zero_one)
            if maximum == one_zero:
                zero_zero_to_be_incresed = maximum - zero_zero
                zero_one_to_be_incresed = maximum - zero_one
                one_one_to_be_incresed = maximum - one_one
                df_zero_zero = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_zero_one = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_one_one = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
                df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
                df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
                df_new = copy.deepcopy(df_zero_zero)
                df_new = df_new.append(df_zero_one)
                df_new = df_new.append(df_one_one)
                df_one_zero = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_new = df_new.append(df_one_zero)
            if maximum == one_one:
                zero_zero_to_be_incresed = maximum - zero_zero
                one_zero_to_be_incresed = maximum - one_zero
                zero_one_to_be_incresed = maximum - zero_one
                df_zero_zero = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_one_zero = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 0)]
                df_zero_one = dataset_orig[(dataset_orig[label] == 0) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
                df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
                df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
                df_new = copy.deepcopy(df_zero_zero)
                df_new = df_new.append(df_one_zero)
                df_new = df_new.append(df_zero_one)
                df_one_one = dataset_orig[(dataset_orig[label] == 1) & (dataset_orig[self.sens_attrs[0]] == 1)]
                df_new = df_new.append(df_one_one) 
            X_train_new, y_train_new = df_new.loc[:, df_new.columns != label], df_new[label]
        else:
            X_train_new, y_train_new = X_train, y_train

        X_test_new = copy.deepcopy(self.X_test)
        if self.remove:
            for sens in self.sens_attrs:
                X_train_new = X_train_new.drop(sens, axis=1)
                X_test_new = X_test_new.drop(sens, axis=1)
        classifier.fit(X_train_new, y_train_new)    
        pred = classifier.predict(X_test_new)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None


        return pred, X_train_new, y_train_new, metric_val, model_name


    def ltdd(self, classifier, do_eval=False):
        """...
        """
        X_train = copy.deepcopy(self.X_train)
        X_test = copy.deepcopy(self.X_test)

        column_train = [column for column in X_train]
        ce_list = []
        ce_times = []
        slope_store = []
        intercept_store = []
        rvalue_store = []
        pvalue_store = []
        column_u = []
        flag = 0
        ce = []
        times = 0

        def Linear_regression(x, slope, intercept):
            return x * slope + intercept

        for i in column_train:
            flag = flag + 1
            if i != self.sens_attrs[0]:
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(X_train[self.sens_attrs[0]], X_train[i])
                rvalue_store.append(rvalue)
                pvalue_store.append(pvalue)
                if pvalue < 0.05:
                    times = times + 1
                    column_u.append(i)
                    ce.append(flag)
                    slope_store.append(slope)
                    intercept_store.append(intercept)
                    X_train[i] = X_train[i] - Linear_regression(X_train[self.sens_attrs[0]], slope, intercept)

        ce_times.append(times)
        ce_list.append(ce)

        X_train2 = X_train.drop([self.sens_attrs[0]],axis = 1)

        for i in range(len(column_u)):
            X_test[column_u[i]] = X_test[column_u[i]] - Linear_regression(X_test[self.sens_attrs[0]], slope_store[i],
                intercept_store[i])

        X_test2 = X_test.drop([self.sens_attrs[0]],axis = 1)

        classifier.fit(X_train2, self.y_train)
        pred = classifier.predict(X_test2)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train, self.y_train, metric_val, "LTDD"
