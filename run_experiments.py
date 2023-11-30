"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import pandas as pd

#For the experiment with two protected attributes, the dataset dictionary has to add the second attribute
#to the sens_attrs list and the corresponding value of the privileged group to the favored tuple.
data_dict = {
    "german": {"sens_attrs": ["sex"], "label": "job", "favored": (0)},
    "compas": {"sens_attrs": ["Race"], "label": "Two_yr_Recidivism", "favored": (0)},
    "communities": {"sens_attrs": ["race"], "label": "crime", "favored": (0)},
    "credit_card_clients": {"sens_attrs": ["sex"], "label": "payment", "favored": (1)},
    "adult_data_set": {"sens_attrs": ["sex"], "label": "salary", "favored": (0)},
    "acs2017_census": {"sens_attrs": ["Race"], "label": "Income", "favored": (0)},
    "implicit10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit20": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit40": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit50": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social20": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social40": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social50": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)}}

"""
PARAMETER SETTINGS START
"""
#Choose the algorithms to test from the following list:
model_list = [
    "Reweighing",\
    "LFR",\
    "DisparateImpactRemover",\
    "Fair-SMOTE",\
    "FairSSL-CT",\
    "FairSSL-LP",\
    "FairSSL-LS",\
    "FairSSL-ST",\
    "LTDD",\
    "PrejudiceRemover",\
    "SquaredDifferenceFairLogistic",\
    "FairnessConstraintModel",\
    "DisparateMistreatmentModel",\
    "ConvexFrameworkModel",\
    "HSICLinearRegression",\
    "GerryFairClassifier",\
    "AdversarialDebiasing",\
    "ExponentiatedGradientReduction",\
    "GridSearchReduction",\
    "MetaFairClassifier",\
    "AdaFair",\
    "FAGTB",\
    "HGR",\
    "FairDummies",\
    "GeneralFairERM",\
    "MultiAdversarialDebiasing",\
    "FairGeneralizedLinearModel",\
    "GradualCompatibility",\
    "RejectOptionClassification",\
    "EqOddsPostprocessing",\
    "CalibratedEqOddsPostprocessing",\
    "JiangNachum",\
    "DPAbstention",\
    "FaX",\
    "FairBayesDPP",\
    "GetFair",\
    "LogisticRegression",\
    "LogisticRegressionRemoved"]

#Choose the randomstate. If set to -1, then a random one is taken. Make sure that all algorithms
#use the same randomstate. Also choose the size of the testdata.
randomstate = -1
testsize = 0.3

#Choose if the parameter optimization component should be activated.
tuning = True
#If tuning is activated: 
#1. If opt = True, then take the optimal parameters from the generated .json file under /configs
#2. If opt = False, then run the grid search parameter tuning technique
#Currently, the config files contain the default values for every algorithm, thus it has to be run
#with opt set to False at least once. Also keep in mind, that for other seeds, other parameter values
#might perform better.
opt = False

#Choose the considered fairness metric: demographic_parity, equalized_odds, treatment_equality, consistency
metric = "demographic_parity"

#Choose if the removal component should be activated
removal = False

#Choose if the binarization component should be activated
binarize = False

#Choose the dataset to run the experiments on (see the dictionary above)
ds = "communities"
sensitive = data_dict[ds]["sens_attrs"]
label = data_dict[ds]["label"]
favored = data_dict[ds]["favored"]

#Choose where to store the files
link = "Results/" + str(metric) + "/" + str(ds) + "/"

try:
    os.makedirs(link)
except FileExistsError:
    pass

"""
PARAMETER SETTINGS END
"""

subprocess.check_call(['python', '-Wignore', 'main.py', '--output', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--testsize', str(testsize), '--randomstate', str(randomstate),
    '--models', str(model_list), '--metric', str(metric), '--tuning', str(tuning),
    '--opt', str(opt), '--binarize', str(binarize), '--removal', str(removal)])

if tuning:
    model_list_eval = []
    for model in model_list:
        model_list_eval.append(model + "_tuned")
    model_list_eval.sort()
else:
    model_list_eval = model_list

subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--folder', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--models', str(model_list_eval), '--metric', str(metric)])
