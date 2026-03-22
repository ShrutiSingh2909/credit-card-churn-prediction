import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load('C:/Users/shruti/Documents/credit-card-churn-prediction/deployment/xgboost_tuned.pkl')

# Load test data
X_test  = np.load('C:/Users/shruti/Documents/credit-card-churn-prediction/data/processed/X_test.npy')
y_test  = np.load('C:/Users/shruti/Documents/credit-card-churn-prediction/data/processed/y_test.npy')

# Load feature names
feature_names = pd.read_csv('C:/Users/shruti/Documents/credit-card-churn-prediction/data/processed/feature_names.csv').iloc[:,0].tolist()

X_test_df = pd.DataFrame(X_test, columns=feature_names)
print("Data loaded!", X_test_df.shape)

# SHAP values
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)
print("SHAP values done!")

# Summary bar plot
plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('C:/Users/shruti/Documents/credit-card-churn-prediction/reports/08_shap_summary_bar.png', dpi=150, bbox_inches='tight')
print("Saved: 08_shap_summary_bar.png")

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False)
plt.tight_layout()
plt.savefig('C:/Users/shruti/Documents/credit-card-churn-prediction/reports/09_shap_beeswarm.png', dpi=150, bbox_inches='tight')
print("Saved: 09_shap_beeswarm.png")

# Waterfall plot
churned_idx = np.where(y_test == 1)[0][0]
explanation = shap.Explanation(
    values        = shap_values[churned_idx],
    base_values   = explainer.expected_value,
    data          = X_test_df.iloc[churned_idx],
    feature_names = feature_names
)
plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig('C:/Users/shruti/Documents/credit-card-churn-prediction/reports/10_shap_waterfall.png', dpi=150, bbox_inches='tight')
print("Saved: 10_shap_waterfall.png")

print("\nAll done! Check reports/ folder for plots.")