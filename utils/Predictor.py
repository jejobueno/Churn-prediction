import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Predictor:
    def __init__(self):
        self.rf_pipe = None
        self.df = pd.DataFrame
        self.ada_pipe = None
        self.svm_pipe = None

    def predict(self, df):
        df_format = pd.DataFrame(columns=self.df.columns)
        df_to_predict = df_format.append(df)

        df_to_predict.fillna(0, inplace=True)
        X = df_to_predict.drop('Attrition_Flag', axis=1).to_numpy()

        return self.rf_pipe.predict(X)

    def trainModel(self, df):
        self.df = df.copy()
        X_train, X_test, y_train, y_test = self.prepareDatasets()
        self.rf_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", RandomForestClassifier(random_state=42))])
        self.rf_pipe.fit(X_train, y_train)

        self.ada_pipe = Pipeline(
            steps=[('scale', StandardScaler()), ("ADA", AdaBoostClassifier(random_state=42, learning_rate=0.7))])
        self.ada_pipe.fit(X_train, y_train)

        # Pipeline of Support Vector Machine using radial basis function kernel
        self.svm_pipe = Pipeline(steps=[('scale', StandardScaler()), ("SVM", SVC(random_state=42, kernel='rbf'))])
        self.svm_pipe.fit(X_train, y_train)

        f1_cross_val_scores = cross_val_score(self.rf_pipe, X_train, y_train, cv=5, scoring='f1')
        ada_val_scores = cross_val_score(self.ada_pipe, X_train, y_train, cv=5, scoring='f1')
        svm_val_scores = cross_val_score(self.svm_pipe, X_train, y_train, cv=5, scoring='f1')
        print(f"""##### Random Forest Score ########
        Cross Validation: {f1_cross_val_scores}
        Score train: {self.rf_pipe.score(X_train, y_train)}
        Score test: {self.rf_pipe.score(X_test, y_test)}""")
        print(f"""##### ADA Boost Score ########
        Cross Validation: {ada_val_scores}
        Score train: {self.ada_pipe.score(X_train, y_train)}
        Score test: {self.ada_pipe.score(X_test, y_test)}""")
        print(f"""##### SVM Score ########
        Cross Validation: {svm_val_scores}
        Score train: {self.svm_pipe.score(X_train, y_train)}
        Score test: {self.svm_pipe.score(X_test, y_test)}""")

        joblib.dump(self, 'utils/model/model.pkl')

    def prepareDatasets(self):
        # Splitting the dataset into train and test
        X = self.df.drop('Attrition_Flag', axis=1).to_numpy()
        y = self.df['Attrition_Flag'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # The percentage of churn samples is 16.07%
        # So let's upsample it
        print('Percentage of churn samples: ', y_train.sum() * 100 / y_train.shape[0])
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        # Checking the percentage again
        print('Percentage of churn samples (upsampled): ', y_train.sum() * 100 / y_train.shape[0])
        # Now we got 50%

        return X_train, X_test, y_train, y_test
