import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load the CSV file
df = pd.read_csv("house_prices.csv")

print("\n===== DATASET =====")
print(df)


# Separate input and output
X = df[["Area_sqft"]]
y = df["Price_lakhs"]


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")


# Predict test data
y_pred = model.predict(X_test)

print("\n===== PREDICTIONS =====")

for actual, predicted in zip(y_test, y_pred):
    print(
        "Actual:",
        actual,
        "Lakhs | Predicted:",
        round(predicted, 2),
        "Lakhs"
    )


# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2 Score :", round(r2, 2))


# Coefficient and intercept
print("\n===== MODEL INFORMATION =====")
print("Coefficient :", model.coef_[0])
print("Intercept :", model.intercept_)


# Predict price for 1100 sqft
new_house = pd.DataFrame({
    "Area_sqft": [1100]
})

predicted_price = model.predict(new_house)

print("\n===== 1100 SQFT PREDICTION =====")
print(
    "Predicted price:",
    round(predicted_price[0], 2),
    "Lakhs"
)


# Plot actual data and regression line
plt.scatter(
    df["Area_sqft"],
    df["Price_lakhs"],
    label="Actual Data"
)

plt.plot(
    df["Area_sqft"],
    model.predict(X),
    label="Regression Line"
)

plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Price Prediction")

plt.legend()
plt.grid()

plt.show()

