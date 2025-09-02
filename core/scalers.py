from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

class StandardScaler:
    def __init__(self):
        self.scaler = SklearnStandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)