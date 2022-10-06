
# Over-sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE

# Under-sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

# Cost-sensitive Classifiers
from imbalanced_ensemble.ensemble import AdaCostClassifier
from imbalanced_ensemble.ensemble import AsymBoostClassifier
from imbalanced_ensemble.ensemble import AdaUBoostClassifier
from sklearn.svm import SVC
from Processing import generate_cost_matrix
from costcla.models import CostSensitiveDecisionTreeClassifier

# Ensemble learning
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from deepforest import CascadeForestClassifier
from balance_algorithm.StackingLR import StackingLogisticRegression
from balance_algorithm.StackingSVM import StackingSVM
from balance_algorithm.StackingDT import StackingDecisionTree

# imbalance-ensemble learning
from imbalanced_ensemble.ensemble import SMOTEBoostClassifier
from imbalanced_ensemble.ensemble import OverBoostClassifier
from imbalanced_ensemble.ensemble import SMOTEBaggingClassifier
from imbalanced_ensemble.ensemble import OverBaggingClassifier
from imbalanced_ensemble.ensemble import RUSBoostClassifier
from imbalanced_ensemble.ensemble import UnderBaggingClassifier
from imbalanced_ensemble.ensemble import EasyEnsembleClassifier
from imbalanced_ensemble.ensemble import BalanceCascadeClassifier
from imbalanced_ensemble.ensemble import BalancedRandomForestClassifier

def data_balancing_ROS(x_train, y_train):
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(x_train, y_train)
    return X, y

def data_balancing_SMOTE(x_train, y_train):
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(x_train, y_train)
    return X, y

def data_balancing_ADASYN(x_train, y_train):
    adasyn = ADASYN(random_state=42)
    X, y = adasyn.fit_resample(x_train, y_train)
    return X, y

def data_balancing_BSMOTE(x_train, y_train):
    bsmote = BorderlineSMOTE(random_state=42)
    X, y = bsmote.fit_resample(x_train, y_train)
    return X, y

def data_balancing_RUS(x_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(x_train, y_train)
    return X, y

def data_balancing_NM(x_train, y_train):
    nm = NearMiss(random_state=42)
    X, y = nm.fit_resample(x_train, y_train)
    return X, y

def data_balancing_CNN(x_train, y_train):
    cnn = CondensedNearestNeighbour(n_neighbors=1, random_state=42)
    X, y = cnn.fit_resample(x_train, y_train)
    return X, y

def data_balancing_TL(x_train, y_train):
    tomeklinks = TomekLinks(random_state=42)
    X, y = tomeklinks.fit_resample(x_train, y_train)
    return X, y

def data_balancing_ENN(x_train, y_train):
    enn = EditedNearestNeighbours(random_state=42)
    X, y = enn.fit_resample(x_train, y_train)
    return X, y

def data_balancing_Adacost(x_train, y_train):
    adacost = AdaCostClassifier(random_state=42)
    adacost.fit(x_train, y_train)
    return adacost

def data_balancing_AsymBoost(x_train, y_train):
    asymboost = AsymBoostClassifier(random_state=42)
    asymboost.fit(x_train, y_train)
    return asymboost

def data_balancing_AdaUBoost(x_train, y_train):
    adauboost = AdaUBoostClassifier(random_state=42)
    adauboost.fit(x_train, y_train)
    return adauboost

def data_balancing_CSSVM(x_train, y_train):
    cssvm = SVC(class_weight='balanced', random_state=42)
    cssvm.fit(x_train, y_train)
    return cssvm

def data_balancing_CSDT(x_train, y_train):
    cost_mat = generate_cost_matrix(y_train)
    csdt = CostSensitiveDecisionTreeClassifier()
    csdt.fit(x_train, y_train, cost_mat)
    return csdt

def data_balancing_Bagging(x_train, y_train):
    bagging = BaggingClassifier(bootstrap=True, random_state=42)
    bagging = bagging.fit(x_train, y_train)
    return bagging

def data_balancing_AdaBoost(x_train, y_train):
    adaboost = AdaBoostClassifier(random_state=42)
    adaboost.fit(x_train, y_train)
    return adaboost

def data_balancing_CatBoost(x_train, y_train):
    catboost = CatBoostClassifier()
    catboost.fit(x_train, y_train)
    return catboost

def data_balancing_XGBoost(x_train, y_train):
    xgboost = XGBClassifier(random_state=42)
    xgboost.fit(x_train, y_train)
    return xgboost

def data_balancing_DeepForest(x_train, y_train):
    deepforest = CascadeForestClassifier(random_state=42)
    deepforest.fit(x_train, y_train)
    return deepforest

def data_balancing_StackingLR(x_train, y_train):
    slr = StackingLogisticRegression(x_train, y_train)
    return slr

def data_balancing_StackingDT(x_train, y_train):
    sdt = StackingDecisionTree(x_train, y_train)
    return sdt

def data_balancing_StackingSVM(x_train, y_train):
    ssvm = StackingSVM(x_train, y_train)
    return ssvm

def data_balancing_SMOTEBoost(x_train, y_train):
    smoteboost = SMOTEBoostClassifier(random_state=42)
    smoteboost.fit(x_train, y_train)
    return smoteboost

def data_balacing_ROSBoost(x_train, y_train):
    rosboost = OverBoostClassifier(random_state=42)
    rosboost.fit(x_train, y_train)
    return rosboost

def data_balacing_SMOTEBagging(x_train, y_train):
    smotebagging = SMOTEBaggingClassifier(random_state=42)
    smotebagging.fit(x_train, y_train)
    return smotebagging

def data_balacing_ROSBagging(x_train, y_train):
    rosbagging = OverBaggingClassifier(random_state=42)
    rosbagging.fit(x_train, y_train)
    return rosbagging

def data_balancing_RUSBoost(x_train, y_train):
    rusboost = RUSBoostClassifier(random_state=42)
    rusboost.fit(x_train, y_train)
    return rusboost

def data_balacing_RUSBagging(x_train, y_train):
    rusbagging = UnderBaggingClassifier(random_state=42)
    rusbagging.fit(x_train, y_train)
    return rusbagging

def data_balancing_EasyEnsemble(x_train, y_train):
    eec = EasyEnsembleClassifier(random_state=42)
    eec.fit(x_train, y_train)
    return eec

def data_balacing_BalanceCascade(x_train, y_train):
    bcc = BalanceCascadeClassifier(random_state=42)
    bcc.fit(x_train, y_train)
    return bcc

def data_balancing_BalancedRandomForest(x_train, y_train):
    brf = BalancedRandomForestClassifier(random_state=42)
    brf.fit(x_train, y_train)
    return brf
