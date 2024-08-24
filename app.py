import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])

# Streamlit application title
st.title('Iris Classification using SVM with Hyperparameter Tuning')

# Interactive selection of hyperparameter tuning approach
approach = st.selectbox('Select Hyperparameter Tuning Approach', 
                        ['Train-Test Split', 'K-Fold Cross Validation', 'GridSearchCV'])

# Set default model
model = None

if approach == 'Train-Test Split':
    st.subheader('Approach: Train-Test Split')
    test_size = st.slider('Test Size', 0.1, 0.5, 0.3)
    kernel = st.selectbox('Kernel', ['linear', 'rbf'])
    C = st.slider('C', 1, 100, 10)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state=42)
    model = svm.SVC(kernel=kernel, C=C, gamma='auto')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.write(f'Overall Model Accuracy: {score:.2f}')

elif approach == 'K-Fold Cross Validation':
    st.subheader('Approach: K-Fold Cross Validation')
    kernel = st.selectbox('Kernel', ['linear', 'rbf'])
    C = st.slider('C', 1, 100, 10)
    scores = cross_val_score(svm.SVC(kernel=kernel, C=C, gamma='auto'), iris.data, iris.target, cv=5)
    st.write(f'Cross-validation scores: {scores}')
    st.write(f'Average score: {np.mean(scores):.2f}')

elif approach == 'GridSearchCV':
    st.subheader('Approach: GridSearchCV')
    param_grid = {
        'C': [1, 10, 20],
        'kernel': ['rbf', 'linear']
    }
    grid_search = GridSearchCV(svm.SVC(gamma='auto'), param_grid, cv=5)
    grid_search.fit(iris.data, iris.target)
    st.write(f'Best Parameters: {grid_search.best_params_}')
    st.write(f'Best Cross-validation Score: {grid_search.best_score_:.2f}')

    results_df = pd.DataFrame(grid_search.cv_results_)
    st.write('Grid Search Results:')
    st.dataframe(results_df[['param_C', 'param_kernel', 'mean_test_score']])

    # Set model to the best estimator from grid search
    model = grid_search.best_estimator_

# Input area for entering feature values
st.subheader('Predict Iris Flower Type')
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

if model:
    if st.button('Predict'):
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        st.write(f'Predicted Iris Flower Type: {iris.target_names[prediction][0]}')
        st.write(f'Model Accuracy: {score:.2f}' if approach == 'Train-Test Split' else '')

# Plotting options
st.subheader('Visualization')
plot_type = st.selectbox('Select Plot Type', ['Scatter Plot', 'Box Plot', 'Pair Plot'])

if st.button('Plot'):
    if plot_type == 'Scatter Plot':
        st.subheader('Scatter Plot of Petal Length vs Petal Width')
        fig, ax = plt.subplots()
        scatter = ax.scatter(df['petal length (cm)'], df['petal width (cm)'], c=pd.Categorical(df['flower']).codes, cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        plt.title('Petal Length vs Petal Width')
        st.pyplot(fig)

    elif plot_type == 'Box Plot':
        st.subheader('Box Plot')
        fig, ax = plt.subplots()
        df.boxplot(ax=ax)
        plt.title('Box Plot of Iris Dataset Features')
        st.pyplot(fig)

    elif plot_type == 'Pair Plot':
        st.subheader('Pair Plot')
        pair_plot = sns.pairplot(df, hue='flower', markers=["o", "s", "D"])
        st.pyplot(pair_plot)
