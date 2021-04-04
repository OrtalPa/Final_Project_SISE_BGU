from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


# removes features with low variance (threshold 0.9)
def get_features_with_high_var(data):
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    res = sel.fit_transform(data)
    print(res.shape)
    return res


def select_k_best(data):
    X_new = SelectKBest(chi2, k=12).fit_transform(data)
    return X_new

