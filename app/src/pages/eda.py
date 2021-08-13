import streamlit as st

def write():
    dataProcessor = st.session_state.dataProcessor

    header = st.beta_container()
    expander = st.beta_expander('Click here to check the df.info()', expanded=False)

    visualize_quantitatives = st.beta_container()
    expander_box = st.beta_expander('Click here to check the box plot and histogram', expanded=False)

    visualize_categoricals = st.beta_container()
    expander_pie = st.beta_expander('Click here to check the pie chart', expanded=False)
    expander_hist = st.beta_expander('Click here to check the histogram', expanded=False)

    correlation = st.beta_container()
    expander_corr = st.beta_expander('Click here to check the correlation matrix', expanded=False)
    model_training = st.beta_container()

    with header:
        st.title('Exploratory Data Analysis (EDA)')
        st.write(' ')
        st.write("After checking that our dataset doesn't have any "
                 "missing values, we start to visualize our data for"
                 " our analysis")

        with expander:
            st.write(""".
                   ---  ------                    --------------  -----
                       Column                    Non-Null Count  Dtype
                   ---  ------                    --------------  -----
                    0   CLIENTNUM                 10127 non-null  int64
                    1   Attrition_Flag            10127 non-null  object
                    2   Customer_Age              10127 non-null  int64
                    3   Gender                    10127 non-null  object
                    4   Dependent_count           10127 non-null  int64
                    5   Education_Level           10127 non-null  object
                    6   Marital_Status            10127 non-null  object
                    7   Income_Category           10127 non-null  object
                    8   Card_Category             10127 non-null  object
                    9   Months_on_book            10127 non-null  int64
                    10  Total_Relationship_Count  10127 non-null  int64
                    11  Months_Inactive_12_mon    10127 non-null  int64
                    12  Contacts_Count_12_mon     10127 non-null  int64
                    13  Credit_Limit              10127 non-null  float64
                    14  Total_Revolving_Bal       10127 non-null  int64
                    15  Avg_Open_To_Buy           10127 non-null  float64
                    16  Total_Amt_Chng_Q4_Q1      10127 non-null  float64
                    17  Total_Trans_Amt           10127 non-null  int64
                    18  Total_Trans_Ct            10127 non-null  int64
                    19  Total_Ct_Chng_Q4_Q1       10127 non-null  float64
                    20  Avg_Utilization_Ratio     10127 non-null  float64
                   dtypes: float64(5), int64(10), object(6)
                   """)

    with visualize_categoricals:
        st.subheader('Explore the categorical values!')
        categorical = st.selectbox('Choose the categorical feature',
                                   dataProcessor.df.select_dtypes(exclude=['float64', 'int64']).columns)
        with expander_pie:
            dataProcessor.plot_pie(categorical)

        with expander_hist:
            dataProcessor.plot_hist(categorical)

    with visualize_quantitatives:
        st.subheader('Now explore the numerical values!')
        quantitative = st.selectbox('Choose the categorical feature',
                                    dataProcessor.df.select_dtypes(include=['float64', 'int64']).columns.drop(
                                        'CLIENTNUM'))

        with expander_box:
            dataProcessor.plot_box(quantitative)

    with correlation:
        st.subheader('Correlation Matrix')
        st.write('Now we can start to check the correlation between our variables'
                 'in a way to see which variables are more correlated with our'
                 'target and prevent to have two or more variables related in between')

        with expander_corr:
            dataProcessor.plot_correlation()
            st.write("""
            Credit limit and Avg_Open_To_Buy are strongly correlated with a value of 0.9. 
            
            The correlation coefficient of Customer age and Months On_book is 0.79,
            which belongs to the category of strong relationship [0.6-0.8]. 

            The correlation coefficient of Total_Trans_Ct and Total_Trans_Amt is 0.88,
             which belongs to the category of extremely strong relationship [0.8-1.0]
             
            So we can remove one of them and keep the feature with the larger Pearson
            coefficient to the target vale. We remove 'Months_on_book', 'Credit_Limit',
            'Total_Trans_Amt'
             """)


