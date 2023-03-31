from tools.logger import Logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder # Use a OnHotEncoder if you can
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt  # for visualization 
import seaborn as sns  # for coloring 
from sklearn import metrics # for evaluation
from time import time


import os


class Spacer:
    def __init__(self) -> None:
        self.logger = Logger(file_log=True)
        self.df_train = pd.read_csv(os.path.abspath(
            os.path.dirname('data')) + "/data/train_titanic.csv")
        self.X_train, self.X_val, self.Y_train, self.Y_val = self.clean_split_dataset(pd.read_csv(os.path.abspath(
            os.path.dirname('data')) + "/data/train_titanic.csv"))
        self.df_test = pd.read_csv(os.path.abspath(
            os.path.dirname('data')) + "/data/test_titanic.csv")

    def model_selection(self):
        """
            ML Classifier Model selection
        """
        try:
            models = [KNeighborsClassifier(3),
                    SVC(kernel="linear", C=0.025),
                    DecisionTreeClassifier(max_depth=5),
                    RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),]
            Y_scores = []
            for i, m in enumerate(models):
                df_fit = m.fit(self.X_train, self.Y_train)
                Y_scores.append({"type": type(m), "score": round(
                    df_fit.score(self.X_val, self.Y_val) * 100, 2)})
            self.logger.info(f" Scores : {Y_scores}")
        except Exception as e:
            raise e
        
    def create_final_dataset(self):
        try:
            self.evaluate_model(self.df_train)
            df = self.df_train.set_index('PassengerId')
            X_before = df.drop(['Transported'], axis=1)
            X = self.__impute_data(X_before)
            print(f"X shape before imputation : {X_before.shape} | X shape after imputation : {X.shape}")
            Y = df['Transported']
            model = RandomForestClassifier(criterion= 'log_loss', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split= 2, n_estimators= 500)
            model.fit(X, Y)
            x_test = self.df_test.set_index('PassengerId')
            process_test = self.__impute_data(x_test)
            print(f"X test shape before imputation : {x_test.shape} | X test shape after imputation : {process_test.shape}")
            ids = x_test.index
            y_pred = model.predict(process_test)
            # creating the dataframe
            df_final = pd.DataFrame(data = y_pred, index = ids, columns=["Transported"])
            df_final.to_csv(os.path.abspath(
                os.path.dirname('data')) + "/data/final_space_titanic.csv")
            self.logger.info(f"Dataset ready ! | Shape : {df_final.shape} ")
        except Exception as e:
            self.logger.error(f"Error occured in the final dataset creation : {e}")

    def evaluate_model(self, df):
        try:
            # Check Correlation
            plt.figure(figsize=(15,15))
            cor = df.corr(numeric_only=True)
            sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
            plt.savefig("correlation_matrix.png")
            plt.show()
            X_train, X_test, Y_train, Y_test = self.clean_split_dataset(df)
            # Build Model
            # Ada Boost Classifier
            # model = AdaBoostClassifier(algorithm= 'SAMME.R', learning_rate= 1.0, n_estimators= 100)
            # Randow Forest Classifier
            model = RandomForestClassifier(criterion= 'log_loss', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split= 2, n_estimators= 500)
            model.fit(X_train, Y_train)

            y_pred = model.predict(X_test)
            cm = metrics.confusion_matrix(Y_test,y_pred)
            ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Oranges')
            ax.set_title('Space Titanic confusion matrix\n\n');
            ax.set_xlabel('\nPredicte if it\'s transported or not')
            ax.set_ylabel('Actual Values');
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['False','True'])
            ax.yaxis.set_ticklabels(['False','True'])
            ## Display the visualization of the Confusion Matrix.
            plt.savefig("confusion_matrix.png")
            plt.show()
            acc = metrics.accuracy_score(Y_test, y_pred)
            self.logger.info(f"The model accuracy score is up to : {round(acc * 100 ,2)} %")
        except Exception as e:
            self.logger.error(f"Error occured in the model evaluation : {e}")

    def clean_split_dataset(self, df):
        """
            Clean and split dataset into training sets
        """
        try:
            df = df.set_index('PassengerId')
            process_df = df.drop(['Transported', 'Name'], axis=1)
            process_df[['cabin_deck', 'cabin_num', 'cabin_side']] = process_df.Cabin.str.split('/', expand=True) if process_df.Cabin is not None else None
            process_df = process_df.drop(['Cabin'], axis=1)
            process_df = self.drop_outliers(process_df)
            categorical_cols = process_df.select_dtypes(include=['object']).columns.values
            numerical_cols = process_df.select_dtypes(include=['float64','int64']).columns.values
            numerical_transformer = SimpleImputer(strategy='median')
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='N/A')),
                ('ordinal', OrdinalEncoder(handle_unknown='error')),
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols),
                ]
            )
            X = preprocessor.fit_transform(process_df)
            Y = df['Transported']

            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.7, random_state=0)

            return X_train, X_val, Y_train, Y_val
        except Exception as e:
            self.logger.error(f"Error occured in the clean and split process : {e}")

    def optimization(self):
        """
            Get Best model parameter
            :return best_params
        """
        try:
            s_time = time()
            # Define X and Y
            df = self.df_train.set_index('PassengerId')
            X = df.drop(['Transported'], axis=1)
            X = self.__impute_data(X)
            Y = df['Transported']
            # Ada Boost Classifier
            # grid = {
            #     'algorithm' : ["SAMME", "SAMME.R"],
            #     'n_estimators' :[10, 50, 100, 500],
            #     'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0],    
            # }
            # DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "auto",max_depth = None)
            # ABC = AdaBoostClassifier(n_estimators = DTC)
            # Random Forest
            model = RandomForestClassifier()
            # # define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
            param_grid = {
                'n_estimators': [10, 50, 100, 500],
                'max_depth': [4,6,8,10],
                'criterion' : ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_leaf': [2,4,6,8,10],
                'min_samples_split': [2,4,6,8,10]
            }
            CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=3)
            CV_rfc.fit(X, Y)
            e_time = time()
            # summarize the best score and configuration
            self.logger.info(f"Optimization execution time {round(e_time - s_time, 2)}s | Best result {CV_rfc.best_score_} | Best Parameters : {CV_rfc.best_params_}")
        except Exception as e:
            self.logger.error(f"Error occured on the optimisation proccess : {e}")

    def drop_outliers(self, df):
        try:
            # Delete feature with high correlation
            cor_matrix = df.corr(numeric_only=True).abs()
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
            process_df = df.drop(to_drop, axis=1) # Drop features gretter than 85% of correlation
            return process_df
        except Exception as e:
            raise e

    def __impute_data(self, df):
        """
            Return imputed data
            :return ndArray
        """
        try:
            # Create new features based o feature Cabin
            df[['cabin_deck', 'cabin_num', 'cabin_side']] = df.Cabin.str.split('/', expand=True) if df.Cabin is not None else None
            df = df.drop(['Cabin', 'Name'], axis=1)
            # Delete feature with high correlation
            df = self.drop_outliers(df)
            # Select categorical and numerical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.values
            numerical_cols = df.select_dtypes(include=['float64','int64']).columns.values
            # Initialize numerical transformer
            numerical_transformer = SimpleImputer(strategy='median')
            # Initialize categorical transformer
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='N/A')),
                ('ordinal', OrdinalEncoder(handle_unknown='error')),
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols),
                ]
            )
            return preprocessor.fit_transform(df)
        except Exception as e:
            self.logger.error(f"Error occured in data imputaion : {e}")
