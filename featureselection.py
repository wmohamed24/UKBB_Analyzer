import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF
from sklearn.feature_selection import chi2

from skfeature.function.similarity_based import fisher_score
from skfeature.function.information_theoretical_based import LCSI


def entropy(vec, base=2):
	"Returns the empirical entropy H(X) in the input vector"
	_, vec = np.unique(vec, return_counts=True)
	prob_vec = np.array(vec/float(sum(vec)))
	if base == 2:
		logfn = np.log2
	elif base == 10:
		logfn = np.log10
	else:
		logfn = np.log
	return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
	"Returns H(X|Y)"
	uy, uyc = np.unique(y, return_counts=True)
	prob_uyc = uyc/float(sum(uyc))
	cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
	return prob_uyc.dot(cond_entropy_x)
	
def mutual_information(x, y):
	"Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y"
	return entropy(x) - conditional_entropy(x, y)

def su_calculation(x, y):
	"Returns 'symmetrical uncertainty' - a symmetric mutual information measure"
	return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))


def chisquare(X, y, n_features):
    '''
    Runs chi-square feature selection on the data (X) and the target values (y) and finds
    the index of the top n_features number of features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    chi2_score, p_val = chi2(X, y)
    index = list(np.argsort(list(chi2_score))[-1*n_features:])
    index.sort()
    return index

def mrmr(X, y, **kwargs):
    """
    This function implements the MRMR feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy= LCSI.lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:
        
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR')
    return F
    

def jmi(X, y, **kwargs):
    """
    This function implements the JMI feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name='JMI', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name='JMI')
    return F


def fcbfHelper(X, y, **kwargs):
    """
    This function implements Fast Correlation Based Filter algorithm

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        delta: {float}
            delta is a threshold parameter, the default value of delta is 0

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    SU: {numpy array}, shape (n_features,)
        symmetrical uncertainty of selected features

    Reference
    ---------
        Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
    """

    n_samples, n_features = X.shape
    if 'delta' in kwargs.keys():
        delta = kwargs['delta']
    else:
        # the default value of delta is 0
        delta = 0

    # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
    t1 = np.zeros((n_features, 2), dtype='object')
    for i in range(n_features):
        f = X[:, i]
        t1[i, 0] = i
        t1[i, 1] = su_calculation(f, y)
    s_list = t1[t1[:, 1] > delta, :]
    # index of selected features, initialized to be empty
    F = []
    # Symmetrical uncertainty of selected features
    SU = []
    while len(s_list) != 0:
        # select the largest su inside s_list
        idx = np.argmax(s_list[:, 1])
        # record the index of the feature with the largest su
        fp = X[:, s_list[idx, 0]]
        np.delete(s_list, idx, 0)
        F.append(s_list[idx, 0])
        SU.append(s_list[idx, 1])
        for i in s_list[:, 0]:
            fi = X[:, i]
            if su_calculation(fp, fi) >= t1[i, 1]:
                # construct the mask for feature whose su is larger than su(fp,y)
                idx = s_list[:, 0] != i
                idx = np.array([idx, idx])
                idx = np.transpose(idx)
                # delete the feature by using the mask
                s_list = s_list[idx]
                length = len(s_list)//2
                s_list = s_list.reshape((length, 2))
    return np.array(F, dtype=int), np.array(SU)


def infogain(X, y, n_features):
    '''
    Runs infogain feature selection on the data (X) and the target values (y) and finds
    the index of the top n_features number of features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    score = mutual_info_classif(X, y,random_state=0)
    index = list(np.argsort(list(score))[-1*n_features:])
    index.sort()
    return index

def reliefF(X, y, n_features):
    '''
    Runs ReliefF algorithm on the data (X) and the target values (y) and finds 
    the index of the top n_features number of features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    fs = ReliefF(n_neighbors=99, n_features_to_keep=n_features)
    fs.fit_transform(X, y)
    index = fs.top_features[:n_features]
    return index

def fcbf(X, y):
    '''
    Runs Fast Correlation-Based Filter feature selection on the data (X) and the target values (y) and finds
    the index of the significant features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    selection = fcbfHelper(X, y)
    index = list(selection[0])
    index.sort()
    return index

def merit_calculation(X, y):
    '''
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    
    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        
    Returns: Merits (float) of a feature subset X
    '''
    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
    '''
    Runs Correlation-based feature selection on the data (X) and the target values (y) and finds
    the index of the significant features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    n_samples, n_features = X.shape
    F = []

    M = []
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 6:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
    return np.array(F)
    
def fisher_exact_test(X, y, n_features):
    '''
    Runs infogain feature selection on the data (X) and the target values (y) and finds
    the index of the top n_features number of features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    
    ranks = fisher_score.fisher_score(X, y)
    index = list(np.argsort(list(ranks))[-1*n_features:])
    index.sort()
    return index

def run_feature_selection(method,X,y):
    '''
    Runs the specific ranking based feature selection method.

    Args:
        method: A string that refers to the feature selection method to be used
        X: An Arraylike structure containing the dataset
        y: An Arraylike structure consisting of the target values

    Returns:
        A list containing the indices of the features that have been selected
    '''
    X=np.array(X)
    y=np.array(y)
    if method[:3] == 'cfs':
        return cfs(X, y)
    elif method[:3] == 'jmi':
        if(len(method.split("_"))==1):
            return jmi(X,y)
        else:
            return jmi(X,y, n_selected_features = int(method.split("_")[1])) 
    elif method[:4] == 'mrmr':
        if(len(method.split("_"))==1):
            return mrmr(X,y)
        else:
            return mrmr(X,y, n_selected_features = int(method.split("_")[1]))    
    elif method[:4] =='fcbf':
        return fcbf(X,y)
    elif method[:7] == 'reliefF':
        return reliefF(X,y,int(method[8:]))
    elif method[:8] == 'infogain':
        return infogain(X,y,int(method[9:]))
    elif method[:6] == 'fisher':
        return fisher_exact_test(X,y,int(method[7:]))
    elif method[:9] == 'chisquare':
        return chisquare(X,y,int(method[10:])) 
    elif method == 'AllFeatures':
        _, cols = X.shape
        return np.arange(cols)