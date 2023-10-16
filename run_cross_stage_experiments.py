"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import copy
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
    "implicit10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social20": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social40": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social50": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)}}

"""
PARAMETER SETTINGS START
"""
#Choose the algorithms to test from the following list:
pre_model_list = ["LTDD", "LFR", "FairSSL-CT", "FairSSL-LP", "FairSSL-LS", "FairSSL-ST", "Fair-SMOTE"]
inpost_model_list = ["MultiAdversarialDebiasing",\
    "AdversarialDebiasing",\
    "GerryFairClassifier",\
    "MetaFairClassifier",\
    "PrejudiceRemover",\
    "ExponentiatedGradientReduction",\
    "GridSearchReduction",\
    "EqOddsPostprocessing",\
    "CalibratedEqOddsPostprocessing",\
    "RejectOptionClassification",\
    "JiangNachum",\
    "AdaFair",\
    "FairGeneralizedLinearModel",\
    "SquaredDifferenceFairLogistic",\
    "FairnessConstraintModel",\
    "DisparateMistreatmentModel",\
    "ConvexFrameworkModel",\
    "HSICLinearRegression",\
    "GeneralFairERM",\
    "FAGTB",\
    "FairDummies",\
    "HGR",\
    "GradualCompatibility",\
    "FaX",\
    "DPAbstention",\
    "GetFair",\
    "FairBayesDPP"]

#Choose the randomstate. If set to -1, then a random one is taken. Make sure that all algorithms
#use the same randomstate. Also choose the size of the testdata.
randomstate = -1
testsize = 0.3

#Choose the considered fairness metric: demographic_parity, equalized_odds, treatment_equality, consistency
metric = "demographic_parity"

#Choose the dataset to run the experiments on (see the dictionary above)
ds = "communities"
sensitive = data_dict[ds]["sens_attrs"]
label = data_dict[ds]["label"]
favored = data_dict[ds]["favored"]

#Choose where to store the files
link = "Results/cross_stages/" + str(metric) + "/" + str(ds) + "/"

try:
    os.makedirs(link)
except FileExistsError:
    pass

"""
PARAMETER SETTINGS END
"""
            
subprocess.check_call(['python', '-Wignore', 'main_cross.py', '--output', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--testsize', str(testsize), '--randomstate', str(randomstate),
    '--pre_models', str(pre_model_list), '--tuning', str(False), '--metric', str(metric),
    '--inpost_models', str(inpost_model_list)])

model_list = []
for pre in pre_model_list:
    for inpost in inpost_model_list:
        model_list.append(pre + "_" + inpost)

subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--folder', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--models', str(model_list), '--metric', str(metric)])
