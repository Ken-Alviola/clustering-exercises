import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import os

from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = ''' select *
                    from predictions_2017 pred
                    inner join (

                                select parcelid, max(transactiondate) as trans_date

                                                from predictions_2017

                                                group by parcelid

                                ) trans on  pred.parcelid = trans.parcelid and pred.transactiondate = trans.trans_date
                                
                    join properties_2017 on pred.parcelid=properties_2017.parcelid    
                   
                    left join airconditioningtype using (airconditioningtypeid)
                    left join `architecturalstyletype` using (`architecturalstyletypeid`)
                    left join `buildingclasstype` using (`buildingclasstypeid`)
                    left join `heatingorsystemtype` using (`heatingorsystemtypeid`)
                    left join `propertylandusetype` using (`propertylandusetypeid`)
                    left join `storytype` using (`storytypeid`)
                    left join `typeconstructiontype` using (`typeconstructiontypeid`)
                    where `transactiondate` between "2017-01-01" and "2017-12-31"
                    and `latitude` is not NULL
                    and `longitude` is not null;
                    '''
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df

def clean_zillow(df):
    '''Takes in a df of zillow data and cleans the data by dropping null values, renaming columns, creating age column, and dealing with             outliers using 1.5x IQR    
    
    return: df, a cleaned pandas dataframe'''
    
    df = df.set_index('parcelid')  

    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = df.dropna()
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet":    
                                    "square_feet","taxamount": "taxes", "taxvaluedollarcnt": "tax_value"})
    
    df['age_in_years'] = 2021 - df.yearbuilt
    df['Bathrooms_cat'] = df.bathrooms.apply(lambda x: "4+" if x >= 4 else x)
    df['Bedrooms_cat'] = df.bathrooms.apply(lambda x: "4+" if x >= 4 else x)
    df['tax_rate'] = round(((df.taxes / df.tax_value) * 100), 2)
    df = df.drop(columns=['yearbuilt']) 
    
    q1 = df.tax_value.quantile(.25)
    q3 = df.tax_value.quantile(.75)
    iqr = q3 - q1
    multiplier = 1.5
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    df = df[df.tax_value > lower_bound]
    df = df[df.tax_value < upper_bound]
    
    return df