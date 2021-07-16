from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
import pandas as pd


def clean_dataset(df):
	# taking only first 5 relevant rows
	df = df.iloc[:, :5]
	# dropping rows with missing info (2021)
	df = df.dropna()
	# show only values for all sub categories
	df = df[df['AUSPRAEGUNG'] == 'insgesamt']
	# remove sum info from the value column
	df = df[df['MONAT'] != 'Summe']
	# make it into descending order chronologically
	df = df[::-1]

	return df


def get_diff(df):
	df['diff'] = df.WERT.diff()
	df = df.dropna()

	return df


def generate_supervised(data):
    supervised_df = data.copy()
    
    supervised_df = supervised_df.drop('WERT', axis=1).drop('JAHR', axis=1).drop('MONAT', axis=1)
    
    #create column for each lag
    for i in range(1,13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['diff'].shift(i)
    
    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    
    return supervised_df


def train_test_split(data):
    data = data.drop(['date'],axis=1)
    train, test = data[0:-12].values, data[-12:].values
    
    return train, test


def scale_data(train_set, test_set):
    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler


def undo_scaling(y_pred, x_test, scaler_obj):  
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
        
    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    
    #inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    
    return pred_test_set_inverted


def predict_df(unscaled_predictions, original_df):
    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(original_df[-13:].date)
    act_sales = list(original_df[-13:].WERT)
    
    for index in range(0,len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)
    
    return df_result



def get_scores(unscaled_df, original_df):
    rmse = np.sqrt(mean_squared_error(original_df.WERT[-12:], unscaled_df.pred_value[-12:]))
    mae = mean_absolute_error(original_df.WERT[-12:], unscaled_df.pred_value[-12:])
    r2 = r2_score(original_df.WERT[-12:], unscaled_df.pred_value[-12:])

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")



def plot_results(results, original_df, model_name):

    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x=original_df.date, y=original_df.WERT, data=original_df, ax=ax, 
                 label='Original', color='mediumblue')
    sns.lineplot(x=results.date, y=results.pred_value, data=results, ax=ax, 
                 label='Predicted', color='Red')
    
    ax.set(xlabel = "Date",
           ylabel = "Accidents",
           title = "Accident Forecasting Prediction")
    
    ax.legend()
    
    sns.despine()