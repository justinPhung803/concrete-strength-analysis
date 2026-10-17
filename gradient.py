import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
df = pd.read_excel(url)
df.columns = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 
              'CoarseAggregate', 'FineAggregate', 'Age', 'Strength']

test_df = df.iloc[500:630].copy()
train_df = pd.concat([df.iloc[:500], df.iloc[630:]]).copy()

X_train = train_df.drop('Strength', axis=1).values
y_train = train_df['Strength'].values
X_test = test_df.drop('Strength', axis=1).values
y_test = test_df['Strength'].values

features = train_df.columns[:-1].tolist()


def mse(y_true, y_pred): 
    return np.mean((y_true - y_pred)**2)

def calculate_r2(y_true, y_pred): 
    return 1 - (mse(y_true, y_pred) / np.var(y_true))


def gradient_descent(X, y, lr=1e-5, epochs=1000):
    n_samples, n_features = X.shape
    X_b = np.c_[np.ones((n_samples, 1)), X]
    theta = np.zeros(n_features + 1)
    history = []
    
    for _ in range(epochs):
        predictions = X_b.dot(theta)
        errors = predictions - y
        gradients = (2/n_samples) * X_b.T.dot(errors)
        theta -= lr * gradients
        history.append(mse(y, predictions))
        
    return theta, history

lr_u = LinearRegression().fit(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_s = LinearRegression().fit(X_train_scaled, y_train)

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
lr_log = LinearRegression().fit(X_train_log, y_train)

summary = []
for name, mod, X_tr, X_te in [
    ("Untransformed", lr_u, X_train, X_test),
    ("Standardized", lr_s, X_train_scaled, X_test_scaled),
    ("Log-Transformed", lr_log, X_train_log, X_test_log)
]:
    p_tr, p_te = mod.predict(X_tr), mod.predict(X_te)
    summary.append({
        "Model": name, 
        "Train MSE": mse(y_train, p_tr), 
        "Test MSE": mse(y_test, p_te),
        "Train R2": calculate_r2(y_train, p_tr), 
        "Test R2": calculate_r2(y_test, p_te)
    })

summary_df = pd.DataFrame(summary)

uni_results = []
for i, f in enumerate(features):
    X_uni = X_train[:, i:i+1]
    lr_uni = LinearRegression().fit(X_uni, y_train)
    uni_results.append({
        "Feature": f, 
        "Univariate Coef": lr_uni.coef_[0], 
        "Multivariate Coef": lr_u.coef_[i]
    })

uni_df = pd.DataFrame(uni_results)

def get_p(X, y):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return model.pvalues[1:]

pvals_df = pd.DataFrame({
    "Feature": features,
    "Untransformed p-value": get_p(X_train, y_train),
    "Scaled p-value": get_p(X_train_scaled, y_train),
    "Log-Transformed p-value": get_p(X_train_log, y_train)
})

print("\n--- 1. Summary Table of Optimal Models ---")
print(summary_df.to_string(index=False))

print("\n--- 2. Univariate vs Multivariate Coefficients ---")
print(uni_df.to_string(index=False))

print("\n--- 3. P-Values Table ---")
print(pvals_df.to_string(index=False))

print("\nRunning Gradient Descent to generate convergence plot...")

theta_u_gd, hist_u = gradient_descent(X_train, y_train, lr=1e-7, epochs=5000)

theta_s_gd, hist_s = gradient_descent(X_train_scaled, y_train, lr=0.1, epochs=500)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_u, color='blue', label='LR = 1e-7')
plt.title('Loss over Iterations: Untransformed')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(hist_s, color='orange', label='LR = 0.1')
plt.title('Loss over Iterations: Standardized (Scaled)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('scaling_convergence.png', dpi=300)
print("Success! 'scaling_convergence.png' has been saved to your folder.")

print("\nRunning Gradient Descent on Log-Transformed Data...")

X_train_log = np.log(X_train + 1)

theta_log_gd, hist_log = gradient_descent(X_train_log, y_train, lr=1e-3, epochs=1500)

plt.figure(figsize=(7, 5))
plt.plot(hist_log, color='green', label='Log-Transformed (LR = 1e-3)')
plt.title('Loss over Iterations: Log(x+1) Transformation')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('log_convergence.png', dpi=300)
print("Success! 'log_convergence.png' has been saved to your folder.")

summary_df.to_csv('model_summary_table.csv', index=False)
print("\nSuccess! 'model_summary_table.csv' has been saved to your folder.")

uni_df.to_csv('univariate_vs_multivariate_coefficients.csv', index=False)
print("Success! 'univariate_vs_multivariate_coefficients.csv' has been saved.")

pvals_df.to_csv('pvalues_table.csv', index=False)
print("Success! 'pvalues_table.csv' has been saved.")

print("\nGenerating Figure 1 and Figure 2 for the Client Report...")

import matplotlib
matplotlib.use('Agg')

plt.figure(figsize=(8, 5))
plt.scatter(train_df['Cement'], train_df['Strength'], alpha=0.6, color='royalblue', edgecolor='k')

plt.xlabel('Cement (kg/m^3)')
plt.ylabel('Compressive Strength (MPa)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_1_cement_vs_strength.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(train_df['Age'], train_df['Strength'], alpha=0.6, color='darkorange', edgecolor='k')

plt.xlabel('Curing Age (Days)')
plt.ylabel('Compressive Strength (MPa)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure_2_age_vs_strength.png', dpi=300)
plt.close()

print("Success! 'figure_1_cement_vs_strength.png' and 'figure_2_age_vs_strength.png' have been saved.")

table_1_stats = train_df.describe().T[['mean', 'std', 'min', 'max']]

print("\n--- Table 1: Descriptive Statistics (Training Set) ---")
print(table_1_stats.to_string())

