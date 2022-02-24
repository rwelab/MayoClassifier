from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
df = pd.read_csv(FILE PATH)
label="Mayo Score"
predictor = TabularPredictor(label=label, path=save_path).fit(train_data, num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, auto_stack = True)
test_data = pd.read_csv(FILE PATH)
y_test = test_data[label]# values to predict
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'], silent=True)
