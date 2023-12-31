"""
main method
"""
import warnings
import argparse
import shelve
import ast
import copy
import math
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import algorithm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.5, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--favored", type=str, help="Tuple of values of privileged group.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--metric", default="mean", type=str, help="Metric which will be used to test\
    the classifier combinations. Default metric: mean.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
parser.add_argument("--pre_models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--inpost_models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
    tuning should be performed. Else, default parameter values are used. Default: False")
args = parser.parse_args()

input_file = args.ds
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
favored = ast.literal_eval(args.favored)
label = args.label
metric = args.metric
randomstate = args.randomstate
if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)
pre_model_list = ast.literal_eval(args.pre_models)
inpost_model_list = ast.literal_eval(args.inpost_models)
if args.tuning == "True":
    tuning = True
else:
    tuning = False

df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
grouped_df = df.groupby(sens_attrs)
group_keys = grouped_df.groups.keys()

privileged_groups = []
unprivileged_groups = []
priv_dict = dict()
unpriv_dict = dict()

if isinstance(favored, tuple):
    for i, fav_val in enumerate(favored):
        priv_dict[sens_attrs[i]] = fav_val
        all_val = list(df.groupby(sens_attrs[i]).groups.keys())
        for poss_val in all_val:
            if poss_val != fav_val:
                unpriv_dict[sens_attrs[i]] = poss_val
else:
    if favored == 0:
        priv_dict[sens_attrs[0]] = 0
        unpriv_dict[sens_attrs[0]] = 1
    elif favored == 1:
        priv_dict[sens_attrs[0]] = 1
        unpriv_dict[sens_attrs[0]] = 0

privileged_groups = [priv_dict]
unprivileged_groups = [unpriv_dict]

#Read the input dataset & split it into training, test & prediction dataset.
#Prediction dataset only needed for evaluation, otherwise size is automatically 0.
X = df.loc[:, df.columns != label]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize,
    random_state=randomstate)

train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
full_dataset = BinaryLabelDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)

train_id_list = []
for i, row in X_train.iterrows():
    train_id_list.append(i)

test_id_list = []
for i, row in X_test.iterrows():
    test_id_list.append(i)

y_train = y_train.to_frame()
y_test = y_test.to_frame()
result_df = copy.deepcopy(y_test)
for sens in sens_attrs:
    result_df[sens] = X_test[sens]

preproc_init = algorithm.Preprocessing(X_train, X_test, y_train, y_test, sens_attrs,
    dataset_train, dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
    index, label, metric, link, input_file, False)
aif_preproc_init = algorithm.AIFPreprocessing(full_dataset, dataset_train, dataset_test, X_test,
    testsize, randomstate, sens_attrs, privileged_groups, unprivileged_groups, label, metric, link, False)

log_regr = LogisticRegression(solver='lbfgs')
log_regr2 = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
dectree = DecisionTreeClassifier()

params = json.load(open('configs/params.json'))
opt_param = dict()

for pre in pre_model_list:
    preproc = copy.deepcopy(preproc_init)
    aif_preproc = copy.deepcopy(aif_preproc_init)
    paramlist = list(params[pre]["default"].keys())
    li = []
    for param in paramlist:
        li.append(params[pre]["default"][param])
    if "LFR" == pre:
        prediction, X_train, y_train, score, pre_model = aif_preproc.lfr(log_regr, k=li[2], Ay=li[0], Az=li[1], do_eval=False)
    elif "Fair-SMOTE" == pre:
        prediction, X_train, y_train, score, pre_model = preproc.smote(log_regr2, do_eval=False)
    elif "FairSSL-ST" == pre:
        prediction, X_train, y_train, score, pre_model = preproc.fairssl(log_regr2, ssl_type=li[0], balancing=li[1]=="True", do_eval=False)
    elif "FairSSL-LS" == pre:
        prediction, X_train, y_train, score, pre_model = preproc.fairssl(log_regr2, ssl_type=li[0], balancing=li[1]=="True", do_eval=False)
    elif "FairSSL-LP" == pre:
        prediction, X_train, y_train, score, pre_model = preproc.fairssl(log_regr2, ssl_type=li[0], balancing=li[1]=="True", do_eval=False)
    elif "FairSSL-CT" == pre:
        prediction, X_train, y_train, score, pre_model = preproc.fairssl(log_regr2, ssl_type=li[0], balancing=li[1]=="True", do_eval=False)
    elif "LTDD" == pre:
        prediction, X_train, y_train, score, pre_model = preproc.ltdd(log_regr2, do_eval=False)
    for inpost in inpost_model_list:
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
        preproc2 = algorithm.Preprocessing(X_train, X_test, y_train, y_test, sens_attrs,
            dataset_train, dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
            index, label, metric, link, input_file, False)
        inproc = algorithm.Inprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored,
            dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, False)
        postproc = algorithm.Postprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored,
            dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, False)
        aif_preproc2 = algorithm.AIFPreprocessing(full_dataset, dataset_train, dataset_test, X_test,
            testsize, randomstate, sens_attrs, privileged_groups, unprivileged_groups, label, metric, link, False)
        aif_inproc = algorithm.AIFInprocessing(dataset_train, dataset_test, sens_attrs,
            privileged_groups, unprivileged_groups, label, metric, link, False)
        aif_postproc = algorithm.AIFPostprocessing(X_train, X_test, y_train, y_test, sens_attrs,
            dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric,
            link, False)
        try:
            paramlist = list(params[inpost]["default"].keys())
            li = []
            for param in paramlist:
                li.append(params[inpost]["default"][param])
            if "AdversarialDebiasing" == inpost:
                prediction, score, model = aif_inproc.adversarial_debiasing("plain_classifier", weight=li[0], debias=li[1]=="True", do_eval=False)
            elif "GerryFairClassifier" == inpost:
                prediction, score, model = aif_inproc.gerryfair(log_regr, gamma=li[0], do_eval=False)
            elif "MetaFairClassifier" == inpost:
                prediction, score, model = aif_inproc.metafair(tau=li[0], do_eval=False)
            elif "PrejudiceRemover" == inpost:
                prediction, score, model = aif_inproc.prejudice_remover(eta=li[0], do_eval=False)
            elif "ExponentiatedGradientReduction" == inpost:
                prediction, score, model = aif_inproc.exponentiated_gradient_reduction(log_regr, eps=li[0], eta=li[1], drop_prot_attr=li[2]=="True", do_eval=False)
            elif "GridSearchReduction" == inpost:
                prediction, score, model = aif_inproc.gridsearch_reduction(log_regr, weight=li[0], drop_prot_attr=li[1]=="True", do_eval=False)
            elif "EqOddsPostprocessing" == inpost:
                prediction, score, model = aif_postproc.eqodds_postproc(log_regr, do_eval=False)
            elif "CalibratedEqOddsPostprocessing" == inpost:
                prediction, score, model = aif_postproc.calibrated_eqodds_postproc(log_regr, do_eval=False)
            elif "RejectOptionClassification" == inpost:
                prediction, score, model = aif_postproc.reject_option_class(log_regr, do_eval=False)
            elif "FaX" == inpost:
                prediction, score, model = postproc.fax(method=li[0], do_eval=False)
            elif "DPAbstention" == inpost:
                prediction, score, model = postproc.dpabst(log_regr, alpha=li[0], do_eval=False)
            elif "GetFair" == inpost:
                prediction, score, model = postproc.getfair(log_regr, lam=li[0], step_size=li[1], episodes=li[2], hidden_size=li[3], layers=li[4], do_eval=False)
            elif "FairBayesDPP" == inpost:
                prediction, score, model = postproc.fair_bayes_dpp(n_epochs=li[0], lr=li[1], batch_size=li[2], n_seeds=li[3], do_eval=False)
            elif "JiangNachum" == inpost:
                prediction, X_train3, y_train3, weights, score, model = postproc.jiang_nachum(log_regr, iterations=li[0], learning=li[1], do_eval=False)
            elif "AdaFair" == inpost:
                prediction, score, model = inproc.adafair(dectree, iterations=li[0], learning_rate=li[1], do_eval=False)
            elif "FairGeneralizedLinearModel" == inpost:
                prediction, score, model = inproc.fglm(lam=li[0], family=li[1], discretization=li[2], do_eval=False)
            elif "SquaredDifferenceFairLogistic" == inpost:
                prediction, score, model = inproc.squared_diff_fair_logistic(lam=li[0], do_eval=False)
            elif "FairnessConstraintModel" == inpost:
                prediction, score, model = inproc.fairness_constraint_model(c=li[0], tau=li[1], mu=li[2], eps=li[3], do_eval=False)
            elif "DisparateMistreatmentModel" == inpost:
                prediction, score, model = inproc.disparate_treatment_model(c=li[0], tau=li[1], mu=li[2], eps=li[3], do_eval=False)
            elif "ConvexFrameworkModel" == inpost:
                prediction, score, model = inproc.convex_framework(lam=li[0], family=li[1], penalty=li[2], do_eval=False)
            elif "HSICLinearRegression" == inpost:
                prediction, score, model = inproc.hsic_linear_regression(lam=li[0], do_eval=False)
            elif "GeneralFairERM" == inpost:
                prediction, score, model = inproc.general_ferm(eps=li[0], k=li[1], do_eval=False)
            elif "FAGTB" == inpost:
                prediction, score, model = inproc.fagtb(estimators=li[0], learning=li[1], lam=li[2], do_eval=False)
            elif "FairDummies" == inpost:
                prediction, score, model = inproc.fair_dummies(batch=li[0], lr=li[1], mu=li[2], second_scale=li[3], epochs=li[4], model_type=li[5], do_eval=False)
            elif "HGR" == inpost:
                prediction, score, model = inproc.hgr(batch=li[0], lr=li[1], mu=li[2], epochs=li[3], model_type=li[4], do_eval=False)
            elif "MultiAdversarialDebiasing" == inpost:
                prediction, score, model = inproc.multi_adv_deb(weight=li[0], do_eval=False)
            elif "GradualCompatibility" == inpost:
                prediction, score, model = inproc.grad_compat(reg=li[0], reg_val=li[1], weights_init=li[2], lambda_=li[3], do_eval=False)            

            result_df[pre_model + "_" + model] = prediction
            result_df.to_csv(link + pre_model + "_" + model + "_prediction.csv")
            result_df = result_df.drop(columns=[pre_model + "_" + model])

        except Exception as e:
            print("------------------")
            print(pre + "_" + inpost)
            print(e)
            print("------------------")
