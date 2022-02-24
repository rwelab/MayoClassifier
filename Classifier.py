#import autogluon and pandas
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

#Read Data
df = pd.read_csv(FILE PATH)
train_data = df
train_data.head()

#Assign Target Label
label="Mayo Score"
print("Summary of class variable: \n", train_data[label].describe())

#Saving models for future use
save_path = 'agModels-BinaryClass',# specifies folder to store trained models

#Model Fitting
predictor = TabularPredictor(label=label, path=save_path).fit(train_data, num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, auto_stack = True)

#Read Test Data
test_data = pd.read_csv(FILE PATH)

y_test = test_data[label]# values to predict
test_data_nolab = test_data.drop(columns=[label])# delete label column to prove we're not cheating
test_data_nolab.head()

predictor = TabularPredictor.load(save_path)# unnecessary, just demonstrates how to load previously-trained predictor from file
y_pred = predictor.predict(test_data_nolab)
print("Predictions:,\n", y_pred)


#Showing all the generated models


print(predictor.get_model_names())
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
results = predictor.fit_summary()
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'], silent=True)

#Calculate Feature Importance
dff=predictor.feature_importance(test_data)
dff.to_csv(FILE PATH)
