import streamlit as st
import pandas as pd
import seaborn as sns
#import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#import time

@st.cache
def readcsv(csv):
    df = pd.read_csv(csv)
    return df


def head(dataframe):
    if len(dataframe) > 1000:
        lenght = 1000
    else:
        lenght = len(dataframe)
    slider = st.slider('Number of rows displayed', 5, lenght)
    st.dataframe(dataframe.head(slider))

@st.cache
def exploratory_func(dataframe):
    desc = dataframe.describe().T
    desc['column'] = desc.index
    exploratory = pd.DataFrame()
    exploratory['NaN'] = dataframe.isnull().sum().values
    exploratory['NaN %'] = 100 * (dataframe.isnull().sum().values / len(dataframe))
    exploratory['NaN %'] = exploratory['NaN %'].apply(lambda x: str(round(x,2)) + " %")
    exploratory['column'] = dataframe.columns
    exploratory['dtype'] = dataframe.dtypes.values
    exploratory = exploratory.merge(desc, on='column', how='left')
    exploratory.loc[exploratory['dtype'] == 'object', 'count'] = len(dataframe) - exploratory['NaN']
    exploratory.set_index('column', inplace=True)
    return exploratory

def heatmap(dataframe, cols = []):
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    check = st.checkbox('Select all', key = 0)
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features:', numcols, default=cols, key = 0)
    if (len(select) > 10):
        annot = False
    else:
        annot = True
    if (len(select) > 1):
        sns.heatmap(dataframe[select].corr(), annot=annot)
        return st.pyplot()

def pairplot(dataframe, cols = []):
    catcols = ['-']
    for i in list(dataframe.select_dtypes(include='object').columns):
        catcols.append(i)
    catcols = tuple(catcols)
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    check = st.checkbox('Select all', key = 1)
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features:', numcols, default=cols, key = 1)
    hue = st.selectbox('Select hue', catcols, key = 0)

    if (len(select) > 1):
        if hue == '-':
            sns.pairplot(dataframe[select])
            return st.pyplot()
        else:
            try:
                copy = select
                copy.append(hue)
                sns.pairplot(dataframe[copy], hue = hue)
                return st.pyplot()
            except:
                st.markdown("An error occurred, please don't use hue for this dataframe")

def boxplot(dataframe):
    numcol = tuple(dataframe.select_dtypes(exclude='object').columns)
    catcol = tuple(dataframe.select_dtypes(include='object').columns)
    select1 = st.selectbox('Selecione a numeric feature', numcol)
    select2 = st.selectbox('Selecione a categorical feature', catcol)
    sns.violinplot(data = dataframe, x = select2, y = select1)
    return st.pyplot()

def scatter(dataframe):
    numcol = tuple(dataframe.select_dtypes(exclude='object').columns)
    select1 = st.selectbox('Select numeric feature', numcol)
    select2 = st.selectbox('Select another numeric feature', numcol)
    plt.scatter(dataframe[select1], dataframe[select2])
    return st.pyplot()

def valuecounts(dataframe):
    select = st.selectbox('Select one feature:', tuple(dataframe.columns))
    st.write(dataframe[select].value_counts())

def input(dataframe, nan_input, feature):
    if nan_input == 'Mean':
        dataframe[feature].fillna(dataframe[feature].mean(), inplace = True)
    elif nan_input == 'Median':
        dataframe[feature].fillna(dataframe[feature].median(), inplace = True)
    elif nan_input == 'Mode':
        dataframe[feature].fillna(dataframe[feature].mode(), inplace = True)
    elif nan_input == 'Zero':
        dataframe[feature].fillna(0, inplace = True)

def drop(dataframe, select):
    if len(select) != 0:
        return dataframe.drop(select, 1)
    else:
        return dataframe

def main():

    st.title('Machine Learning initial analysis')
    st.image('ia.jpg', use_column_width=True)
    st.markdown('Upload a csv file, use the sidebar to pre-processing and EDA, and feel free to make a '
                'prediction using one of the available models.')
    file = st.file_uploader('Upload your csv file', type='csv')
    if file is not None:
        st.sidebar.subheader('Visualization')
        sidemulti = st.sidebar.multiselect('Plots: ',('Heatmap','Pairplot','Violinplots','Scatterplot'))

        df0 = pd.DataFrame(readcsv(file))
        st.sidebar.subheader('Drop columns:')
        sidedrop = st.sidebar.multiselect('Columns to be dropped: ', tuple(df0.columns))
        df = drop(df0, sidedrop)
        st.sidebar.subheader('Fill missing values')
        sidemiss = st.sidebar.selectbox('Feature: ', tuple(df.columns))
        sidemethod = st.sidebar.selectbox('Method: ', ('Mean','Median','Mode','Zero'))
        input_missing = st.sidebar.button('Fill')
        if input_missing:
            input(df, sidemethod, sidemiss)

        st.header('Dataframe Visualization')
        head(df)
        st.header('Descritive Statistics')
        st.dataframe(exploratory_func(df))
        st.header('Value Counts')
        valuecounts(df)

        if ('Heatmap' in sidemulti):
            st.header('Heatmap')
            st.subheader('Select numeric features:')
            heatmap(df)

        if ('Pairplot' in sidemulti):
            st.header('Pairplot')
            st.subheader('Select numeric features and 1 categorical feature at most')
            pairplot(df)

        if ('Violinplots' in sidemulti):
            st.header('Select features for the Violinplot')
            boxplot(df)
        if('Scatterplot') in sidemulti:
            st.header('Select features for the Scatterplot')
            scatter(df)

        st.header('Random Forest:')
        model = st.selectbox('Select the target:', ('Regressor','Classifier'))
        target = st.selectbox('Select the target:', tuple(df.columns))
        X = pd.get_dummies(df.drop(target,1))
        y = df[target]
        tt_slider = st.slider('% Size of test split:', 1,99)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01*tt_slider, random_state=42)

        predict = st.button('Predict')
        if predict:
            st.header('Results:')
            st.subheader('Score:')
            if model == 'Regressor':
                rf = RandomForestRegressor()
            elif model == 'Classifier':
                rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            st.markdown(rf.score(X_test, y_test))
            #plt.figure(num=None, figsize=(6, 4), facecolor='w', edgecolor='k')
            feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
            st.subheader('Feature Importances:')
            feat_importances.nlargest(10).plot(kind='barh', figsize = (8,8))
            st.pyplot()
if __name__ == '__main__':
    main()
