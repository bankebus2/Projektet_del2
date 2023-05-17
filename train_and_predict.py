from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

def plot_predictions(bensin_params, diesel_params, el_params, elhybrid_params, laddhybrid_params, etanol_params, title):
    plt.scatter(bensin_params[0], bensin_params[1], color=bensin_params[2], label=bensin_params[3])
    plt.scatter(diesel_params[0], diesel_params[1], color=diesel_params[2], label=diesel_params[3])
    plt.scatter(el_params[0], el_params[1], color=el_params[2], label=el_params[3])
    plt.scatter(elhybrid_params[0], elhybrid_params[1], color=elhybrid_params[2], label=elhybrid_params[3])
    plt.scatter(laddhybrid_params[0], laddhybrid_params[1], color=laddhybrid_params[2], label=laddhybrid_params[3])
    plt.scatter(etanol_params[0], etanol_params[1], color=etanol_params[2], label=etanol_params[3]) 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.show()


# Fetch dataframe created in analyze_data.ipynb
complete_df = pickle.load(open('./complete_df.pkl', 'rb'))

X = complete_df[['E85', 'Diesel_fuel', 'BF95','Elpris']].values



look_ahead = 3  # Offset for y. Most likely the effects of current fuel prices will be seen a few months later.

if look_ahead == 0:     # X and y the same month
    print('Same month')
    y_bensin = complete_df['Bensin_andel'].values
    y_diesel = complete_df['Diesel_andel'].values
    y_el = complete_df['El_andel'].values
    y_elhybrid = complete_df['Elhybrid_andel'].values
    y_laddhybrid = complete_df['Laddhybrid_andel'].values
    y_etanol = complete_df['Etanol_andel'].values
    

else:   # Look x months ahead
    print(f"{look_ahead} months ahead")
    
    y_bensin = complete_df['Bensin_andel'].shift(-look_ahead).values
    y_diesel = complete_df['Diesel_andel'].shift(-look_ahead).values
    y_el = complete_df['El_andel'].shift(-look_ahead).values
    y_elhybrid = complete_df['Elhybrid_andel'].shift(-look_ahead).values
    y_laddhybrid = complete_df['Laddhybrid_andel'].shift(-look_ahead).values
    y_etanol = complete_df['Etanol_andel'].shift(-look_ahead).values
    
    # Adjust in order not to end up outside of df.
    X = X[:-look_ahead]
    
    y_bensin = y_bensin[:-look_ahead]
    y_diesel = y_diesel[:-look_ahead]
    y_el = y_el[:-look_ahead]
    y_elhybrid = y_elhybrid[:-look_ahead]
    y_laddhybrid = y_laddhybrid[:-look_ahead]
    y_etanol = y_etanol[:-look_ahead]


X_train, X_test, y_train_bensin, y_test_bensin = train_test_split(X, y_bensin, test_size=0.1, random_state=42)
X_train, X_test, y_train_diesel, y_test_diesel = train_test_split(X, y_diesel, test_size=0.1, random_state=42)
X_train, X_test, y_train_el, y_test_el = train_test_split(X, y_el, test_size=0.1, random_state=42)
X_train, X_test, y_train_elhybrid, y_test_elhybrid = train_test_split(X, y_elhybrid, test_size=0.1, random_state=42)
X_train, X_test, y_train_laddhybrid, y_test_laddhybrid = train_test_split(X, y_laddhybrid, test_size=0.1, random_state=42)
X_train, X_test, y_train_etanol, y_test_etanol = train_test_split(X, y_etanol, test_size=0.1, random_state=42)


# Linear Regression
skk_linear_bensin = LinearRegression()
skk_linear_diesel = LinearRegression()
skk_linear_el = LinearRegression()
skk_linear_elhybrid = LinearRegression()
skk_linear_laddhybrid = LinearRegression()
skk_linear_etanol = LinearRegression()

# Train
skk_linear_bensin.fit(X_train, y_train_bensin)
skk_linear_diesel.fit(X_train, y_train_diesel)
skk_linear_el.fit(X_train, y_train_el)
skk_linear_elhybrid.fit(X_train, y_train_elhybrid)
skk_linear_laddhybrid.fit(X_train, y_train_laddhybrid)
skk_linear_etanol.fit(X_train, y_train_etanol)


# Predict
linear_predictions_bensin = skk_linear_bensin.predict(X_test)
linear_predictions_diesel = skk_linear_diesel.predict(X_test)
linear_predictions_el = skk_linear_el.predict(X_test)
linear_predictions_elhybrid = skk_linear_elhybrid.predict(X_test)
linear_predictions_laddhybrid = skk_linear_laddhybrid.predict(X_test)
linear_predictions_etanol = skk_linear_etanol.predict(X_test)


# Decision Tree
dtree_bensin = DecisionTreeRegressor()
dtree_diesel = DecisionTreeRegressor()
dtree_el = DecisionTreeRegressor()
dtree_elhybrid = DecisionTreeRegressor()
dtree_laddhybrid = DecisionTreeRegressor()
dtree_etanol = DecisionTreeRegressor()

# Train
dtree_bensin.fit(X_train, y_train_bensin)
dtree_diesel.fit(X_train, y_train_diesel)
dtree_el.fit(X_train, y_train_el)
dtree_elhybrid.fit(X_train, y_train_elhybrid)
dtree_laddhybrid.fit(X_train, y_train_laddhybrid)
dtree_etanol.fit(X_train, y_train_etanol)


# Predict
dtree_predictions_bensin = dtree_bensin.predict(X_test)
dtree_predictions_diesel = dtree_diesel.predict(X_test)
dtree_predictions_el = dtree_el.predict(X_test)
dtree_predictions_elhybrid = dtree_elhybrid.predict(X_test)
dtree_predictions_laddhybrid = dtree_laddhybrid.predict(X_test)
dtree_predictions_etanol = dtree_etanol.predict(X_test)

# Random Forest
rf_bensin = RandomForestRegressor()
rf_diesel = RandomForestRegressor()
rf_el = RandomForestRegressor()
rf_elhybrid = RandomForestRegressor()
rf_laddhybrid = RandomForestRegressor()
rf_etanol = RandomForestRegressor()

# Train
rf_bensin.fit(X_train, y_train_bensin)
rf_diesel.fit(X_train, y_train_diesel)
rf_el.fit(X_train, y_train_el)
rf_elhybrid.fit(X_train, y_train_elhybrid)
rf_laddhybrid.fit(X_train, y_train_laddhybrid)
rf_etanol.fit(X_train, y_train_etanol)


# Predict
rf_predictions_bensin = rf_bensin.predict(X_test)
rf_predictions_diesel = rf_diesel.predict(X_test)
rf_predictions_el = rf_el.predict(X_test)
rf_predictions_elhybrid = rf_elhybrid.predict(X_test)
rf_predictions_laddhybrid = rf_laddhybrid.predict(X_test)
rf_predictions_etanol = rf_etanol.predict(X_test)

# Plot results

# Linear Regression
plot_predictions((y_test_bensin, linear_predictions_bensin, 'blue', 'Bensin_andel'),(y_test_diesel, linear_predictions_diesel, 'green', 'Diesel_andel'),\
                 (y_test_el, linear_predictions_el, 'red', 'El_andel'),(y_test_elhybrid, linear_predictions_elhybrid, 'purple', 'Elhybrid_andel'), \
                    (y_test_laddhybrid, linear_predictions_laddhybrid, 'black', 'Laddhybrid_andel'),\
                        (y_test_etanol, linear_predictions_etanol, 'yellow', 'Etanol_andel'), 'Linear Regression Predictions')



# Decision Tree
plot_predictions((y_test_bensin, dtree_predictions_bensin, 'blue', 'Bensin_andel'),(y_test_diesel, dtree_predictions_diesel, 'green', 'Diesel_andel'),\
                 (y_test_el, dtree_predictions_el, 'red', 'El_andel'),(y_test_elhybrid, dtree_predictions_elhybrid, 'purple', 'Elhybrid_andel'), \
                    (y_test_laddhybrid, dtree_predictions_laddhybrid, 'black', 'Laddhybrid_andel'),\
                        (y_test_etanol, dtree_predictions_etanol, 'yellow', 'Etanol_andel'),'Decision Tree Predictions')

# Random Forest
plot_predictions((y_test_bensin, rf_predictions_bensin, 'blue', 'Bensin_andel'),(y_test_diesel, rf_predictions_diesel, 'green', 'Diesel_andel'),\
                 (y_test_el, rf_predictions_el, 'red', 'El_andel'),(y_test_elhybrid, rf_predictions_elhybrid, 'purple', 'Elhybrid_andel'), \
                    (y_test_laddhybrid, rf_predictions_laddhybrid, 'black', 'Laddhybrid_andel'),\
                        (y_test_etanol, rf_predictions_etanol, 'yellow', 'Etanol_andel'), 'Random Forest Predictions')





