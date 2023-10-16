"""
In this python file multiple classification models are trained.
"""
import numpy as np
import pandas as pd
import copy
import torch
import torch.optim as optim
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from .FaX_AI import FaX_methods
from .dpabst.post_process import TransformDPAbstantion
from .GetFair.OptimizerModel import MetaOptimizer, MetaOptimizerMLP, MetaOptimizerDirection
from .GetFair.Optimizer import TrainerEpisodic, TrainerEpisodicMultiple
from .GetFair.Rewardfunctions import stat_parity, diff_FPR, diff_FNR, diff_FPR_FNR, diff_Eoppr, diff_Eodd
from .FairBayesDPP.dataloader import FairnessDataset
from .FairBayesDPP.algorithm import FairBayes_DPP
from .jiang_nachum.label_bias import LabelBiasDP_helper, LabelBiasEOD_helper, LabelBiasEOP_helper
from .evaluation.eval_classifier import acc_bias


class Postprocessing():
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
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, favored,
        dataset_train, dataset_test, privileged_groups, unprivileged_groups, label,
        metric, link, remove):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_tr_np = self.y_train.to_numpy().reshape((self.y_train.to_numpy().shape[0],))
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.label = label
        self.metric = metric
        self.link = link
        self.remove = remove


    def fax(self, method="MIM", do_eval=False):
        """
        ...
        """
        X2 = self.X_train.loc[:, self.X_train.columns != self.sens_attrs[0]].to_numpy()
        Z2 = self.X_train[self.sens_attrs[0]].to_frame().to_numpy()
        Y2 = self.y_train.to_numpy()

        X3 = self.X_test.loc[:, self.X_test.columns != self.sens_attrs[0]].to_numpy()
        Z3 = self.X_test[self.sens_attrs[0]].to_frame().to_numpy()
        Y3 = self.y_test.to_numpy()

        if method == "MIM":
            model = FaX_methods.MIM(X2, Z2, Y2)
        elif method == "OPT":
            model = FaX_methods.Optimization(X2, Z2, Y2, influence="shap")

        pred = model.predict(X3)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "FaX"


    def dpabst(self, classifier, alpha, do_eval=False):
        """
        ...
        """
        X_train = self.X_train[[c for c in self.X_train if c not in self.sens_attrs] + self.sens_attrs].to_numpy()
        X_test = self.X_test[[c for c in self.X_test if c not in self.sens_attrs] + self.sens_attrs].to_numpy()

        Cs = np.logspace(-4, 4, 30)
        parameters = dict()
        parameters["C"] = Cs
        clf = GridSearchCV(LogisticRegression(solver='liblinear'), parameters, cv=5, refit=True)
        clf.fit(X_train, self.y_train.to_numpy())

        alphas = {0: alpha, 1: alpha}
        transformer = TransformDPAbstantion(clf, alphas)
        transformer.fit(X_train)
        pred = transformer.predict(X_test)
        for i, p in enumerate(pred):
            if p == 10000:
                pred[i] = 0

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "DPAbstention"


    def getfair(self, classifier, lam=0.55, step_size=0.04, episodes=5, hidden_size=30, layers=1, do_eval=False):
        """
        ...
        """
        if self.metric == "demographic_parity":
            metric = stat_parity
        elif self.metric == "equal_opportunity":
            metric = diff_Eoppr
        elif self.metric == "equalized_odds":
            metric = diff_Eodd
        elif self.metric == "treatment_equality":
            metric = diff_FPR_FNR
        else:
            metric = stat_parity


        class Dataset():
            def __init__(self, tr_dt, tr_s, tr_c, ts_dt, ts_s, ts_c, vl_dt, vl_s, vl_c):
                self.tr_dt = tr_dt
                self.tr_s = tr_s
                self.tr_c = tr_c
                self.ts_dt = ts_dt
                self.ts_s = ts_s
                self.ts_c = ts_c
                self.vl_dt = vl_dt
                self.vl_s = vl_s
                self.vl_c = vl_c
        
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=42)
        A_train = X_train[self.sens_attrs]
        A_val = X_val[self.sens_attrs]
        A_test = self.X_test[self.sens_attrs]
        X_train = X_train[[c for c in X_train if c not in self.sens_attrs]].to_numpy()
        X_val = X_val[[c for c in X_val if c not in self.sens_attrs]].to_numpy()
        X_test = self.X_test[[c for c in self.X_test if c not in self.sens_attrs]].to_numpy()
        if len(self.sens_attrs) == 1:
            A_train = A_train.to_numpy().reshape((A_train.to_numpy().shape[0],))
            A_val = A_val.to_numpy().reshape((A_val.to_numpy().shape[0],))
            A_test = A_test.to_numpy().reshape((A_test.to_numpy().shape[0],))
        else:
            A_train = A_train.to_numpy()
            A_val = A_val.to_numpy()
            A_test = A_test.to_numpy()
        y_train = y_train.to_numpy().reshape((y_train.to_numpy().shape[0],))
        y_val = y_val.to_numpy().reshape((y_val.to_numpy().shape[0],))
        y_test = self.y_test.to_numpy().reshape((self.y_test.to_numpy().shape[0],))
        classifier.fit(X_train, y_train)

        data = Dataset(X_train, A_train, y_train, X_test, A_test, y_test, X_val, A_val, y_val)

        meta = MetaOptimizerDirection(hidden_size=hidden_size, layers=layers, output_size=2)
        trainer = TrainerEpisodic(meta, classifier, data, metric, classifiertype="linear")
        pred, pred2 = trainer.train(accuracy_threshold=lam, step_size=step_size, episodes=episodes)
        #res = trainer.get_best_parameters()

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "GetFair"


    def jiang_nachum(self, classifier, iterations, learning, do_eval=False):
        """
        ...
        """
        if self.metric == "demographic_parity":
            LB = LabelBiasDP_helper()
        elif self.metric == "equalized_odds":
            LB = LabelBiasEOD_helper()
        elif self.metric == "equal_opportunity":
            LB = LabelBiasEOP_helper()
        else:
            LB = LabelBiasDP_helper()

        protected_train = [np.array(self.dataset_train.convert_to_dataframe()[0][g]) for g in self.sens_attrs]
        protected_test = [np.array(self.dataset_test.convert_to_dataframe()[0][g]) for g in self.sens_attrs]
        multipliers = np.zeros(len(protected_train))
        if self.metric in ("equalized_odds", "equal_opportunity"):
            weights = np.array([1] * self.X_train.shape[0])
        if self.metric == ("equalized_odds"):
            multipliers = np.zeros(len(protected_train) * 2)
        weights = np.array([1] * self.X_train.shape[0])

        X_train = copy.deepcopy(self.X_train)
        if self.remove:
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
        X_tr_np = X_train.to_numpy()

        for it in range(iterations):
            if self.metric not in ("equalized_odds", "equal_opportunity"):
                weights = LB.debias_weights(self.y_tr_np, protected_train, multipliers)
            clf = copy.deepcopy(classifier)
            clf.fit(X_tr_np, self.y_tr_np, weights)
            prediction = clf.predict(X_tr_np)
            if self.metric in ("equalized_odds", "equal_opportunity"):
                weights = LB.debias_weights(self.y_tr_np, prediction, protected_train, multipliers)

            acc, violations, pairwise_violations = LB.get_error_and_violations(prediction, self.y_tr_np, protected_train)
            multipliers += learning * np.array(violations)

        X_train = copy.deepcopy(self.X_train)
        X_test = copy.deepcopy(self.X_test)
        if self.remove:
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)
        classifier.fit(X_train, self.y_train, weights)
        pred = classifier.predict(X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, self.X_train, self.y_train, weights, metric_val, "JiangNachum"


    def fair_bayes_dpp(self, n_epochs=200, lr=1e-1, batch_size=512, n_seeds=5, do_eval=False):
        """
        ...
        """
        from .FairBayesDPP.models import Classifier
        n_layers = 3
        n_hidden_units = 32

        device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu')
        resultprop_PP = pd.DataFrame()
        resultprop_PP_base = pd.DataFrame()

        for seed in range(n_seeds):
            seed = seed * 5
            # Set a seed for random number generation
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            dataset = FairnessDataset(dataset="Custom", device=device, X_train=self.X_train,
                y_train=self.y_train, X_test=self.X_test, y_test=self.y_test, sens_attrs=self.sens_attrs)
            dataset.normalize()
            input_dim = dataset.XZ_train.shape[1]

            # Create a classifier model
            net = Classifier(n_layers=n_layers, n_inputs=input_dim, n_hidden_units=n_hidden_units)
            net = net.to(device)

            # Set an optimizer
            optimizer = optim.Adam(net.parameters(), lr=lr)

            # Fair classifier training
            eta1, eta0, t1_pp, t0_pp = FairBayes_DPP(dataset=dataset,dataset_name="Custom",
                net=net, optimizer=optimizer, device=device, n_epochs=n_epochs,
                batch_size=batch_size, seed=seed)

        pred = []
        eta1_counter = 0
        eta0_counter = 0
        for i, row in self.X_test.iterrows():
            if row[self.sens_attrs[0]] == 1:
                #pred.append(1) if eta1[eta1_counter] >= t1_pp else pred.append(0)
                pred.append(1) if eta1[eta1_counter] >= 0.5 else pred.append(0)
                eta1_counter += 1
            else:
                #pred.append(1) if eta0[eta0_counter] >= t0_pp else pred.append(0)
                pred.append(1) if eta0[eta0_counter] >= 0.5 else pred.append(0)
                eta0_counter += 1

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "FairBayesDPP"
