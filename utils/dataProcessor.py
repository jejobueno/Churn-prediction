import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as stl


class DataProcessor:
    def __init__(self):
        self.df = pd.read_csv('data/BankChurners.csv')

    def preprocess(self):
        #self.analyze()
        # Deleting features with strong correlation between each other
        correlations = self.df.corr()
        correlated_features = set()

        for i in range(len(correlations.columns)):
            for j in range(i):
                if abs(correlations.iloc[i, j]) > 0.8:
                    colname = correlations.columns[i]
                    correlated_features.add(colname)

        self.df.drop(correlated_features, axis=1, inplace=True)

        # Checking the correlation matrix again
        plt.figure()
        correlations = self.df.corr()
        fig = sns.heatmap(correlations, center=0, annot=True, cmap="YlGnBu")
        plt.tight_layout()
        stl.pyplot(fig)

        # Encode our target value 'Attrition_Flag'
        self.df.Attrition_Flag.replace({'Attrited Customer': 1, 'Existing Customer': 0}, inplace=True)
        self.df.Gender.replace({'F': 1, 'M': 0}, inplace=True)
        to_dummies_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

        for feature in to_dummies_features:
            cv_dummies = pd.get_dummies(self.df[feature])
            self.df = pd.concat([self.df, cv_dummies], axis=1)
            del self.df[feature]

    def analyze(self):
        # Check if there is any nan values
        print('Missing values:', self.df.isna().sum().sum())

        # Now let's see the correlation:
        correlations = self.df.corr()
        sns.heatmap(correlations, center=0, annot=True, cmap="YlGnBu")
        plt.tight_layout()
        plt.show()

        # Checking normality is variables
        features = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Dependent_count',
                    'Avg_Utilization_Ratio', 'Months_Inactive_12_mon', 'Total_Trans_Amt', 'Credit_Limit']
        for feature in features:
            fig, axes = plt.subplots(2, 1)
            sns.boxplot(x=self.df[feature], showmeans=True, ax=axes[0]).set_title('Box Plot')
            sns.histplot(x=self.df[feature], ax=axes[1]).set_title('Histogram')
            plt.tight_layout()
            fig.suptitle('Analyzing ' + feature)
            stl.pyplot()

        # Checking at categorical features
        features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Attrition_Flag']
        for feature in features:
            fig, axes = plt.subplots(1, 2)
            data = self.df.groupby(feature)['CLIENTNUM'].count()
            data.plot.pie(autopct="%.1f%%", ax=axes[0])
            sns.histplot(x=self.df[feature], ax=axes[1]).set_title('Histogram')
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.suptitle('Analyzing ' + feature)
            stl.pyplot()
