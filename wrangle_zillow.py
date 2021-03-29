import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
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
    sql_query = '''select *
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
        df = pd.read_csv('zillow.csv', index_col=1)
        
    return df

def handle_missing_values(df, prop_required_row = 0.70, prop_required_col = 0.5):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df

def clean_zillow(df):
    '''Takes in a df of zillow data and cleans the data by dropping null values, renaming columns, creating age column, and dealing with outliers using 1.5x IQR    
    
    return: df, a cleaned pandas dataframe'''
    
    #renaming duplicate IDs and dropping
    df.columns = ['typeconstructiontypeid',
                  'storytypeid',
                  'propertylandusetypeid',
                  'heatingorsystemtypeid',
                  'buildingclasstypeid',
                  'architecturalstyletypeid',
                  'airconditioningtypeid',
                  'id',
                  'parcelid',
                  'logerror',
                  'transactiondate',
                  'parcelid2',
                  'trans_date',
                  'id2',
                  'parcelid3',
                  'basementsqft',
                  'bathroomcnt',
                  'bedroomcnt',
                  'buildingqualitytypeid',
                  'calculatedbathnbr',
                  'decktypeid',
                  'finishedfloor1squarefeet',
                  'calculatedfinishedsquarefeet',
                  'finishedsquarefeet12',
                  'finishedsquarefeet13',
                  'finishedsquarefeet15',
                  'finishedsquarefeet50',
                  'finishedsquarefeet6',
                  'fips',
                  'fireplacecnt',
                  'fullbathcnt',
                  'garagecarcnt',
                  'garagetotalsqft',
                  'hashottuborspa',
                  'latitude',
                  'longitude',
                  'lotsizesquarefeet',
                  'poolcnt',
                  'poolsizesum',
                  'pooltypeid10',
                  'pooltypeid2',
                  'pooltypeid7',
                  'propertycountylandusecode',
                  'propertyzoningdesc',
                  'rawcensustractandblock',
                  'regionidcity',
                  'regionidcounty',
                  'regionidneighborhood',
                  'regionidzip',
                  'roomcnt',
                  'threequarterbathnbr',
                  'unitcnt',
                  'yardbuildingsqft17',
                  'yardbuildingsqft26',
                  'yearbuilt',
                  'numberofstories',
                  'fireplaceflag',
                  'structuretaxvaluedollarcnt',
                  'taxvaluedollarcnt',
                  'assessmentyear',
                  'landtaxvaluedollarcnt',
                  'taxamount',
                  'taxdelinquencyflag',
                  'taxdelinquencyyear',
                  'censustractandblock',
                  'airconditioningdesc',
                  'architecturalstyledesc',
                  'buildingclassdesc',
                  'heatingorsystemdesc',
                  'propertylandusedesc',
                  'storydesc',
                  'typeconstructiondesc']
    
    df = df.drop(columns=['parcelid2','parcelid3','id2'])
    
    #df = df.set_index('parcelid')  
    #subsetting single unit properties
    df = df[(df.propertylandusetypeid.isin(['261','262','264','266','263','268','273','276','275','279','260'])) | (df.unitcnt == 1)]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull()) & (df.calculatedfinishedsquarefeet>350)]
    
    #dropping nulls
    df = handle_missing_values(df)
    #df['age_in_years'] = 2021 - df.yearbuilt     
    
    
    #getting rid of erroneous zipcodes
    df = df[(df.regionidzip < 100000)&(df.regionidzip.notnull())]
    
    #removing records of less than 100 nulls
    df = df[df.taxvaluedollarcnt.notnull()]
    df = df[df.taxamount.notnull()]
    df = df[df.landtaxvaluedollarcnt.notnull()]
    
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    cols_to_remove = ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc']
    
    df = df.drop(columns=cols_to_remove)
    
    df.heatingorsystemdesc.fillna('None', inplace = True)
    df.unitcnt.fillna(1, inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)
    
    # fill missing values for buildingqualitytypeid with median value
    df.buildingqualitytypeid.fillna(6.0, inplace = True)
    df = df.dropna()
    
    #outlier removal 1.5x IQR
    q1 = df.taxvaluedollarcnt.quantile(.25)
    q3 = df.taxvaluedollarcnt.quantile(.75)
    iqr = q3 - q1
    multiplier = 1.5
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    df = df[df.taxvaluedollarcnt > lower_bound]
    df = df[df.taxvaluedollarcnt < upper_bound]
    
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet":    
                                    "square_feet","taxamount": "taxes", "taxvaluedollarcnt": "tax_value"})
    return df

def split_zillow(df, stratify_by=None):
    """
    train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(df, test_size=.3, random_state=123)
    else:
        train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate[stratify_by])
    
    return train, validate, test


def wrangle_zillow(split=False):
    '''
    wrangle_zillow will read zillow.csv as a pandas dataframe,
    clean the data
    split the data
    return: train, validate, test sets of pandas dataframes from zilow if split = True
    '''
    df = clean_zillow(get_zillow_data())
    if split == True:
        return split_zillow(df)
    else:
        return df
    
def scale_data(train,validate,test):
    '''Accepts train, validate, test data frames and applies standard scaler
    return: train, validate, test scaled pandas dataframe'''
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    return train_scaled, validate_scaled, test_scaled
    
#functions below are for tabling nulls
def missing_rows(df):
    missing = pd.DataFrame()
    missing['num_rows_missing'] = df.isnull().sum()
    missing['pct_rows_missing'] = df.isnull().sum()/len(df) * 100
    return missing.sort_values('pct_rows_missing',ascending=False)    

def missing_cols(df):
    missing_col= pd.DataFrame(df.isnull().sum(axis =1).value_counts(),columns=['num_rows']) 
    missing_col = missing_col.reset_index()
    missing_col = missing_col.rename(columns={'index':'num_cols_missing'})
    missing_col['pct_cols_missing'] = missing_col.num_cols_missing / len(df.columns) * 100
    return missing_col.sort_values('pct_cols_missing',ascending=False)

def missing_zero_values_table(df):
    '''This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values
        and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    null_count = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
    mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
    mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table

