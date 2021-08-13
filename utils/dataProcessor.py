import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


class DataProcessor:
    def __init__(self):
        self.df_org = pd.read_csv('data/BankChurners.csv')
        # According with the specifications we delete the las two columns Naive Bayes
        self.df_org = self.df_org.drop([
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
            axis=1)
        self.df = self.df_org.copy()
        self.df_processed = pd.DataFrame()

    def preprocess(self):
        self.analyze()
        df = self.df

        df.drop(['CLIENTNUM'], axis=1, inplace=True)
        df.Attrition_Flag.replace({'Attrited Customer': 1, 'Existing Customer': 0}, inplace=True)
        df.Gender.replace({'F': 1, 'M': 0}, inplace=True)

        # Cheking the correlation matrix and the spearman correlation to delete
        # most correlated features
        correlations = df.corr()
        spearman_correlation = df.corr(method='spearman')

        # Here we are going to select the variables with the stronger relation
        # in between to after select which one to drop
        upper_tri = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
        to_drop_row = [upper_tri.index[upper_tri[feature] == correlation][0] for feature in to_drop for correlation in
                       upper_tri[feature] if correlation > 0.7]

        for i in range(len(to_drop)):
            if spearman_correlation['Attrition_Flag'][to_drop_row[i]] > spearman_correlation['Attrition_Flag'][
                to_drop[i]]:
                df.drop(to_drop[i], axis=1, inplace=True)
            else:
                df.drop(to_drop_row[i], axis=1, inplace=True)

        # Encode our categorical values
        cv_dummies = pd.get_dummies(df)

        return cv_dummies

    def analyze(self):
        # Check if there is any nan values
        print('Missing values:', self.df.isna().sum().sum())

    def plot_pie(self, feature: str):
        fig = plt.figure()
        data = self.df.groupby(feature).size()
        data.plot.pie(autopct="%.1f%%", pctdistance=0.5).set_title(feature + ' pie chart')
        st.pyplot(fig)

    def plot_hist(self, feature: str):
        fig = plt.figure()
        sns.histplot(x=self.df[feature], hue=self.df['Attrition_Flag']).set_title(feature + ' Histogram')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    def plot_box(self, feature):
        fig, axes = plt.subplots(2, 1)
        sns.boxplot(x=self.df[feature], showmeans=True, ax=axes[0]).set_title(feature + ' Box Plot')
        sns.histplot(x=self.df[feature], hue=self.df['Attrition_Flag'], ax=axes[1]).set_title(feature + 'Histogram')
        plt.tight_layout()
        st.pyplot(fig)

    def plot_correlation(self):
        df = self.df.copy()
        df.drop(['CLIENTNUM'], axis=1, inplace=True)
        df.Attrition_Flag.replace({'Attrited Customer': 1, 'Existing Customer': 0}, inplace=True)
        df.Gender.replace({'F': 1, 'M': 0}, inplace=True)

        matrix = np.triu(df.corr())
        correlations = df.corr()

        fig = plt.figure(figsize=(8, 6), dpi=80)
        sns.heatmap(correlations, cmap="YlGnBu", annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, mask=matrix,
                    cbar=False)
        st.pyplot(fig)

        spearman_correlation = df.corr(method='spearman')
        fig = plt.figure(figsize=(8, 6), dpi=80)
        sns.heatmap(spearman_correlation, cmap="YlGnBu", annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, mask=matrix,
                    cbar=False)
        st.pyplot(fig)


