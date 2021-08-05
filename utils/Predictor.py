from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Predictor:
    def __init__(self, df):
        self.df = df

    def predict(self):
        self.trainModel()


    def trainModel(self):
        X_train, X_test, y_train, y_test = self.prepareDatasets()
        rf_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", RandomForestClassifier(random_state=42))])
        rf_pipe.fit(X_train, y_train)

        ada_pipe = Pipeline(
            steps=[('scale', StandardScaler()), ("ADA", AdaBoostClassifier(random_state=42, learning_rate=0.7))])
        ada_pipe.fit(X_train, y_train)

        # Pipeline of Support Vector Machine using radial basis function kernel
        svm_pipe = Pipeline(steps=[('scale', StandardScaler()), ("SVM", SVC(random_state=42, kernel='rbf'))])
        svm_pipe.fit(X_train, y_train)

        f1_cross_val_scores = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring='f1')
        print(f"""##### Random Forest Score ########
        Cross Validation: {f1_cross_val_scores}
        Score train: {rf_pipe.score(X_train, y_train)}
        Score test: {rf_pipe.score(X_test, y_test)}""")
        ada_f1_cross_val_scores = cross_val_score(ada_pipe, X_train, y_train, cv=5, scoring='f1')
        print(f"""##### Ada Boost Classifier ########
        Cross Validation: {ada_f1_cross_val_scores}
        Score train: {ada_pipe.score(X_train, y_train)}
        Score test: {ada_pipe.score(X_test, y_test)}""")
        svm_f1_cross_val_scores = cross_val_score(svm_pipe, X_train, y_train, cv=5, scoring='f1')
        print(f"""##### SVM ########
        Cross Validation: {svm_f1_cross_val_scores}
        Score train: {svm_pipe.score(X_train, y_train)}
        Score test: {svm_pipe.score(X_test, y_test)}""")

    def prepareDatasets(self):
        # Splitting the dataset into train and test
        X = self.df.drop('Attrition_Flag', axis=1).to_numpy()
        y = self.df['Attrition_Flag'].to_numpy().reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # The percentage of churn samples is 16.07%
        # So let's upsample it
        print(y_train.sum()*100/y_train.shape[0])
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        # Checking the percentage again
        print(y_train.sum()*100/y_train.shape[0])
        # Now we got 50%

        return X_train, X_test, y_train, y_test
