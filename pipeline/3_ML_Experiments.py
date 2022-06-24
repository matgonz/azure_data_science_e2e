# Databricks notebook source
# MAGIC %md
# MAGIC # ML Experiments
# MAGIC 
# MAGIC So, the last we used AutoML to create a baseline model. Now, we'll to analyse metrics, the bias variance tradeoff and propose the best solution for this problem.
# MAGIC 
# MAGIC **AutoML**
# MAGIC - [MLflow experiments](#mlflow/experiments/713584551057122/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - [Best Model](#mlflow/experiments/713584551057122/runs/de7019603a78477ab6391adf466f1a80)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Libraries

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import collections
import numpy as np

from matplotlib import pyplot

import mlflow
from mlflow.tracking import MlflowClient

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
import databricks.automl_runtime

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer
#from sklearn import set_config

from lightgbm import LGBMClassifier

#set_config(display="diagram")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Load Data
# MAGIC 
# MAGIC We'll download the artifacts created by AutoML Experiments to use here.

# COMMAND ----------

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)

# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("8a47aae0f21648beaa8df758e517c597", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

aux = df_loaded['Churn'].value_counts()
aux = aux / aux.sum()
print(aux)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Functions
# MAGIC 
# MAGIC Defining preprocessors of pipeline and experiment functions

# COMMAND ----------

feature_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
       'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
       'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Computer',
       'PreferredLoginDevice_Mobile_Phone', 'PreferredLoginDevice_Phone',
       'CityTier_1', 'CityTier_2', 'CityTier_3', 'PreferredPaymentMode_CC',
       'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash_on_Delivery',
       'PreferredPaymentMode_Credit_Card', 'PreferredPaymentMode_Debit_Card',
       'PreferredPaymentMode_E_wallet', 'PreferredPaymentMode_UPI',
       'Gender_Female', 'Gender_Male', 'PreferedOrderCat_Fashion',
       'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_&_Accessory',
       'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile_Phone',
       'PreferedOrderCat_Others', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single']

features_to_numeric = ["WarehouseToHome", "CashbackAmount", "DaySinceLastOrder", "Tenure", "OrderAmountHikeFromlastYear"]

def transform_features(dataframe, features):
    
    # Select features
    dataframe = dataframe[features]
    
    # Cast to numeric values
    for col in features:
        if col in features_to_numeric:        
            dataframe[col] = pd.to_numeric(dataframe[col])
        
    # Mean and Standard deviation
    np.random.seed(10)
    std_scaler = StandardScaler()
    result_scaler = std_scaler.fit_transform(dataframe[features])

    return result_scaler


def classification_metrics(y_true, y_pred, _target_names):
    """Function to help calculating classification metrics"""
    print(classification_report(y_true, y_pred, _target_names))
    print('Accuracy Score: ', accuracy_score(y_true, y_pred))
    print('Balanced Accuracy Score: ', balanced_accuracy_score(y_true, y_pred))

    
def generate_bootstrap_sample(dataframe, target_column, frac_items_to_return=0.5):
    """
        Description: Generator bootstrap sample.
        Params:
        - dataframe
        - target_column
        - frac_items_to_return
    """
    np.random.seed(10)
    
    df_aux = dataframe.copy()
    
    target_classes = df_aux[target_column].unique()
    
    sample_df_list = [df_aux[df_aux[target_column] == target_class].sample(frac=frac_items_to_return) for target_class in target_classes]
        
    return pd.concat(sample_df_list)


def experiment_runner(_subsample_sets, feature_subset, target_colname):
    """Helper function to run MLflow experiment on a feature subset"""
    
    with mlflow.start_run() as run:
    
        "Build Subsets of Features"
        experimental_data_subsets = [transform_features(sample_set[feature_subset], feature_subset) for sample_set in subsample_sets]
        targets = [sample_set[target_colname] for sample_set in _subsample_sets]

        "Fit on each subset using Cross-Validation"
        experimental_scores = []

        for features, target in zip(experimental_data_subsets, targets):
            
            np.random.seed(10)
            cv=5
            scoring='balanced_accuracy'
            clf = LGBMClassifier(objective='binary')

            gs = GridSearchCV(clf, scoring=scoring, param_grid={}, cv=5)
            gs.fit(features, target)

            # score = gs.cv_results_
            score = gs.cv_results_["mean_test_score"][0]
            experimental_scores.append(score)

        "Record experiment results"
        #mlflow.log_param("len_features_subset", len(feature_subset))
        mlflow.log_param("len_features_subset", feature_subset)
        
        mlflow.log_metric("mean score", np.mean(experimental_scores))
        mlflow.log_metric("std score", np.std(experimental_scores))
        
        
def feature_selection(X, y, features):
    
    model = RandomForestClassifier()
    model.fit(X, y)
    # get importance
    feature_importance = pd.DataFrame({'features': features, 'importance': model.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=True, inplace=True)
    
    pyplot.figure(figsize=(10,10))
    pyplot.barh(feature_importance['features'], feature_importance['importance'])
    pyplot.title('Feature Importance by RandomForestClassifier')
    pyplot.show()
    
    print(feature_importance.describe())
    
    return feature_importance

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 4. Split out train and test data
# MAGIC 
# MAGIC - Training Dataset: 80%
# MAGIC - Test Dataset: 20%

# COMMAND ----------

# Target column
target_col = "Churn"
# Features
split_X = df_loaded.drop([target_col], axis=1)
# Target Series
split_y = df_loaded[target_col]

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(split_X, split_y, train_size=0.8, random_state=591423154, stratify=split_y)

# Check distribution
dist_arr = []
arr_y = [('training', y_train), ('test', y_test)]
 
for dataset_name, dist in arr_y:
    c = collections.Counter(dist)
    total = c[0] + c[1]
    dist_arr.append({'dataset':dataset_name, '0': c[0] / total, '1': c[1] / total})
 
pd.DataFrame(dist_arr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 5. Baseline
# MAGIC 
# MAGIC Split training set into training and validation:
# MAGIC - Train (80% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)

# COMMAND ----------

##### Hyperparameters used by AutoML #####
# colsample_bytree=0.5438762801993433,
# lambda_l1=2.80383773385808,
# lambda_l2=5.401710572056857,
# learning_rate=0.03280149487316273,
# max_bin=415,
# max_depth=9,
# min_child_samples=80,
# n_estimators=1499,
# num_leaves=205,
# path_smooth=16.164096261021655,
# subsample=0.7728650113993107,
# random_state=591423154

# COMMAND ----------

X_train_baseline, X_val_baseline, y_train_baseline, y_val_baseline = train_test_split(X_train, y_train, train_size=0.8, random_state=591423154, stratify=y_train)

dist_arr = []
arr_y = [('training', y_train_baseline), ('test', y_val_baseline)]
 
for dataset_name, dist in arr_y:
    c = collections.Counter(dist)
    total = c[0] + c[1]
    dist_arr.append({'dataset':dataset_name, '0': c[0] / total, '1': c[1] / total})
 
pd.DataFrame(dist_arr)

# COMMAND ----------

with mlflow.start_run() as run:

    np.random.seed(10)

    X_train_baseline_scaled = transform_features(dataframe=X_train_baseline, features=feature_cols)
    X_val_baseline_scaled = transform_features(dataframe=X_val_baseline, features=feature_cols)

    objective = 'binary'
    lgbmc_classifier = LGBMClassifier(objective=objective)
    lgbmc_classifier.fit(X_train_baseline_scaled, y_train_baseline)

    y_val_baseline_pred = lgbmc_classifier.predict(X_val_baseline_scaled)

    classification_metrics(y_true=y_val_baseline, y_pred=y_val_baseline_pred, _target_names=[0,1])
    
    # Log params
    mlflow.log_param("objective", objective)
    
    # Log model
    mlflow.sklearn.log_model(lgbmc_classifier, "LGBMClassifier")
    
    # Log metrics
    mlflow.log_metric('Accuracy Score Validation Set', accuracy_score(y_val_baseline, y_val_baseline_pred))
    mlflow.log_metric('Balanced Accuracy Score Validation Set', balanced_accuracy_score(y_val_baseline, y_val_baseline_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC Access MLflow runs associated with this notebook

# COMMAND ----------

results = mlflow.search_runs()

# Select metrics
results = results[~results['metrics.Accuracy Score Validation Set'].isnull()]
results = results[~results['tags.mlflow.log-model.history'].isnull()]
results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference [TODO]
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

model_name = "Churn"

model_uri = f"runs:/{ mlflow_run.info.run_id }/model"

model_name, model_uri

# COMMAND ----------

registered_model_version = mlflow.register_model(model_uri, model_name)
