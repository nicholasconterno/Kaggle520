import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
"""
get_accounts() - returns a dataframe of accounts.csv and does some preprocessing
"""
def get_accounts():
    df = pd.read_csv('account.csv',encoding='ISO-8859-1')
    #drop columns that are not needed
    df.drop(['shipping.zip.code','shipping.city','relationship'], axis=1, inplace=True)
    #convert first donated to datetime
    df['billing.zip.code'] = df['billing.zip.code'].astype(str)
    #convert billing zip code to everything before the -
    df['billing.zip.code'] = df['billing.zip.code'].str.split('-').str[0]
    return df
"""
get_concerts_2014_2015() - returns a dataframe of concerts_2014_15.csv
"""
def get_concerts_2014_2015():
    df = pd.read_csv('concerts_2014_15.csv')
    return df
"""
get_concerts() - returns a dataframe of concerts.csv
"""
def get_concerts():
    df = pd.read_csv('concerts.csv')
    # Update the season column to keep only the first year
    df['season'] = df['season'].str.split('-').str[0].astype(int)
    return df
"""
get_zipcodes() - returns a dataframe of zipcodes.csv and does some preprocessing
"""
def get_zipcodes():
    df = pd.read_csv('zipcodes.csv')
    #convert zipcodes to strings
    df['Zipcode'] = df['Zipcode'].astype(str)
    #drop decommissioned column
    df.drop(columns='Decommisioned', inplace=True)
    return df
"""
get_train() - returns a dataframe of train.csv
"""
def get_train():
    df = pd.read_csv('train.csv')
    return df
"""
get_test() - returns a dataframe of test.csv
"""
def get_test():
    df = pd.read_csv('test.csv')
    return df
"""
get_subscriptions() - returns a dataframe of subscriptions.csv alonf with some preprocessing
"""
def get_subscriptions():
    df = pd.read_csv('subscriptions.csv')
    #convert season to everything before the -
    df['season'] =df['season'].str.split('-').str[0]
    #convert multiple.subs to numeric
    df['multiple.subs'] = df['multiple.subs'].replace({'yes': 1, 'no': 0})
    #make a pivot table of the subscriptions on subscription tier for each account.id and season
    pivot_df = df.pivot_table(index='account.id', columns='season', values='subscription_tier', aggfunc='count',fill_value=0).reset_index()
    df.rename(columns={'location': 'orchestra_location'}, inplace=True)  # Rename the column to avoid ambiguity'})
    #fill in the missing values with 0
    for col in pivot_df.columns:
        if col != 'account.id':  # skip the account.id column
            pivot_df[col] = pivot_df[col].fillna(0)

    return pivot_df
def get_tickets():
    tickets = pd.read_csv('tickets_all.csv')

    # Update the season column to keep only the first year
    tickets['season'] = tickets['season'].str.split('-').str[0].astype(int)
    return tickets
"""
merge_accounts_with_zipcodes() - returns a dataframe of accounts.csv and zipcodes.csv merged together
We also fill some nan values as a preprocessing step
"""
def merge_accounts_with_zipcodes_and_subscriptions():
    #merge accounts with zipcodes
    df_merged = pd.merge(get_accounts(), get_zipcodes(), left_on="billing.zip.code", right_on="Zipcode", how="left")
    #fill in missing values with the mode
    df_merged['first.donated'] = df_merged['first.donated'].fillna(df_merged['first.donated'].mode()[0])
    #merge with subscriptions
    subs = get_subscriptions()
    df_merged = pd.merge(df_merged, subs, on="account.id", how="left")
    #fill in missing values
    columns_to_fill_with_median = [
    'Lat',
    'Long',
    'TaxReturnsFiled',
    'EstimatedPopulation',
    'TotalWages'
    ]
    # Calculate the median values for these columns
    medians = df_merged[columns_to_fill_with_median].median()
    # Fill in the missing values with the medians
    df_merged.fillna(medians, inplace=True)
    # For other columns, continue with the 'Unknown' and other specific fill values
    other_fill_values = {
    'amount.donated.2013':0,
    'amount.donated.lifetime':0,
    'no.donations.lifetime':0,
    'no.seats':0,
    'shipping.zip.code': 'Unknown',
    'shipping.city': 'Unknown',
    'billing.city': 'Unknown',
    'relationship': 'Unknown',
    'first.donated': 'Unknown',
    'season': 'Unknown',
    'package': 'Unknown',
    'orchestra_location': 'Unknown',
    'section': 'Unknown',
    'price.level': 0,
    'subscription_tier': 0.0,
    'multiple.subs': 'no',
    'Zipcode': 'Unknown',
    'ZipCodeType': 'Unknown',
    'City': 'Unknown',
    'State': 'Unknown',
    'LocationType': 'Unknown',
    'Location': 'Unknown'
    }
    df_merged.fillna(other_fill_values, inplace=True)
    

    return df_merged
"""
fill_missing_values() - returns a dataframe with missing values filled in
"""
def fill_missing_values(df):
    object_cols = df.select_dtypes(include=['object']).columns
    number_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Fill missing values in object columns with 'unknown'
    for col in object_cols:
        df[col].fillna('unknown', inplace=True)

    # Fill missing values in numeric columns with median of that column
    for col in number_cols:
        df[col].fillna(df[col].median(), inplace=True)

    return df
"""
merge_tickets_and_concerts() - returns a dataframe of tickets_all.csv and concerts.csv merged together
Also does some preprocessing

"""
def merge_tickets_and_concerts():
    tickets_and_concerts = pd.merge(get_tickets(), get_concerts(), on=['season', 'location','set'], how='left')
    tickets_and_concerts = fill_missing_values(tickets_and_concerts)
    return tickets_and_concerts
"""
final_merge() - returns a dataframe of all the dataframes we will be using merged together (does not include train.csv or test.csv)
Also does some more preprocessing
"""
def final_merge():
    # Extract the necessary columns from tickets_and_concerts
    columns_to_merge = [
    'account.id', 'price.level', 'no.seats', 
    'marketing.source', 'location', 
    'multiple.tickets', 'concert.name'
    ]
    tickets_and_concerts_subset = merge_tickets_and_concerts()[columns_to_merge]
    # drop duplicates
    tickets_and_concerts_subset= tickets_and_concerts_subset.drop_duplicates(subset=['account.id'], keep='last')
    # Merge the accounts with zipcodes and subscriptions with the tickets and concerts
    final_df = pd.merge(merge_accounts_with_zipcodes_and_subscriptions(), tickets_and_concerts_subset, on='account.id', how='left')

    # Define the newly added non-numerical columns
    non_numerical_columns = ['marketing.source','location', 'multiple.tickets', 'concert.name']

    for col in non_numerical_columns:
        final_df[col] = final_df[col].fillna('Unknown')
    return final_df

"""
haversine_distance() - returns the distance between two points on the globe
"""
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
"""
add_engineered_features() - returns a dataframe with engineered features added
"""
def add_engineered_features():
    df_merged = final_merge()
    # Add a column for whether the account has donated
    df_merged['donated'] = 0
    df_merged.loc[df_merged['amount.donated.lifetime'] > 0, 'donated'] = 1
    # Coordinates for the center of San Francisco
    sf_lat, sf_lon = 37.7749, -122.4194
    # Calculate distance from San Francisco to each account holder
    df_merged['distance_from_orchestra'] = df_merged.apply(
    lambda row: haversine_distance(sf_lat, sf_lon, row['Lat'], row['Long']), axis=1)
    #calculate average subscription tier over the last 5 years
    df_merged['avg.subscription_tier'] = (df_merged['2009']+df_merged['2010']+df_merged['2011']+df_merged['2012']+df_merged['2013'])/5
    #replace marketing source with average target value
    # Calculate mean of target per category
    mean_encode = df_merged.groupby('marketing.source')['2013'].mean().to_dict()
    # Map the encoded values to the 'marketing.source' column
    df_merged['marketing.source_encoded'] = df_merged['marketing.source'].map(mean_encode)
    # Calculate frequency of each category
    frequency_encode = df_merged['location'].value_counts(normalize=True).to_dict()

    # Map the encoded values to the 'location' column
    df_merged['location_encoded'] = df_merged['location'].map(frequency_encode)

    #binary subscription tier for the last five years
    df_merged['2013_binary'] = df_merged['2013'].apply(lambda x: 1 if x > 0 else 0)
    df_merged['2012_binary'] = df_merged['2012'].apply(lambda x: 1 if x > 0 else 0)
    df_merged['2011_binary'] = df_merged['2011'].apply(lambda x: 1 if x > 0 else 0)
    df_merged['2010_binary'] = df_merged['2010'].apply(lambda x: 1 if x > 0 else 0)
    df_merged['2009_binary'] = df_merged['2009'].apply(lambda x: 1 if x > 0 else 0)

    #sum of binary subscription tiers
    df_merged['sum.binary'] = df_merged['2013_binary']+df_merged['2012_binary']+df_merged['2011_binary']+df_merged['2010_binary']+df_merged['2009_binary']
    return df_merged

"""
get_merged_dataframe() - returns a dataframe of all the dataframes we will be using merged together (does not include train.csv or test.csv)
"""
def get_merged_dataframe():
    df_merged = add_engineered_features()
    return df_merged

def get_train_test():
    train_df = get_train()
    #merge train with merged dataframe
    df_merged = get_merged_dataframe()
    df = pd.merge(df_merged, train_df, on="account.id", how="inner")
    #take the labels off and make them our target variable
    X =  df.drop('label', axis=1)
    y = df['label']
    # Label encode categorical columns
    label_encoders = {}
    # Loop through the columns to label encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        # Skip the account.id column
        if col != 'account.id':
            # Initialize label encoder
            le = LabelEncoder()
            # Fit label encoder on the column
            X[col] = le.fit_transform(X[col].astype(str))
            # Save the fitted label encoder to the dictionary
            label_encoders[col] = le
    # Split the data into train and validation sets
    unique_ids = X['account.id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2,shuffle=True)
    
    # Use the split IDs to create train and validation dataframes
    X_train = X[X['account.id'].isin(train_ids)]
    y_train = y[X['account.id'].isin(train_ids)]
    X_test = X[X['account.id'].isin(test_ids)]
    y_test = y[X['account.id'].isin(test_ids)]
    X_train = X_train.drop('account.id', axis=1)
    X_test = X_test.drop('account.id', axis=1)
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, label_encoders


def train_model(X_train,y_train):
    # Train a CatBoostClassifier
    catboost = CatBoostClassifier(
        iterations=100,
        learning_rate=0.02,
        verbose=200,
        eval_metric='AUC', # Optimizing for AUC directly
        auto_class_weights='Balanced', # Handling class imbalance
    )
    # Fit the model
    catboost.fit(X_train, y_train, use_best_model=True)
    return catboost

def generate_predictions(model,X_test):
    # Generate predictions
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions

def get_roc_score(y_test,predictions):
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, predictions)
    return roc_auc

def save_test_predictions(trainedModel, scaler, label_encoders):
    test = get_test()
    # merge test with merged dataframe
    df_merged = get_merged_dataframe()
    # rename test column 'ID' to 'account.id'
    test.rename(columns={'ID': 'account.id'}, inplace=True)
    df = pd.merge(df_merged, test, on="account.id", how="inner")
    
    # Drop the 'account.id' column for prediction
    df.drop('account.id', axis=1, inplace=True)
    
    # Transform categorical columns using label encoders
    for col in df.select_dtypes(include=['object']).columns:
        # Skip the 'account.id' column
        if col != 'account.id':
            # Get the corresponding label encoder
            le = label_encoders[col]
            
            # Create a set of unique labels encountered in the training data
            unique_labels = set(le.classes_)
            
            # Transform the column using the label encoder and handle unknown labels
            encoded_values = df[col].apply(lambda x: le.transform([x])[0] if x in unique_labels else -1)
            
            df[col] = encoded_values
    # Scale the data using the same scaler used during training
    df = scaler.transform(df)
    
    # Generate predictions
    predictions = generate_predictions(trainedModel, df)
    
    # Create a DataFrame with 'account.id' and 'Predicted' columns
    result_df = pd.DataFrame({'ID': test['account.id'], 'Predicted': predictions})
    
    # Save the predictions to a CSV file
    result_df.to_csv('final_predictions.csv', index=False)


def main():
    # get training and testing data
    X_train, X_test, y_train, y_test, scaler,label_encoders = get_train_test()
    # train model
    model = train_model(X_train,y_train)
    # generate predictions
    predictions = generate_predictions(model,X_test)
    # get roc score
    roc_auc = get_roc_score(y_test,predictions)
    print(roc_auc)
    return model,scaler,label_encoders


if __name__ == "__main__":
    model,scaler,label_encoders = main()
    #save test predictions (commented out for cleanliness)
    #save_test_predictions(model,scaler,label_encoders)

