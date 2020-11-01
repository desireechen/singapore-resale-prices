import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

def fetch_data(data_path):
    df = pd.read_csv(data_path)
    return df

def regroup_storey(df):
    df.loc[df['storey_range'] == '01 TO 03', 'storey'] = '1-low'
    df.loc[df['storey_range'] == '04 TO 06', 'storey'] = '1-low'
    df.loc[df['storey_range'] == '07 TO 09', 'storey'] = '2-mid'
    df.loc[df['storey_range'] == '10 TO 12', 'storey'] = '2-mid'
    df.loc[df['storey_range'] == '13 TO 15', 'storey'] = '3-high'
    df.loc[df['storey_range'] == '16 TO 18', 'storey'] = '3-high'
    df.loc[df['storey_range'] == '19 TO 21', 'storey'] = '4-higher'
    df.loc[df['storey_range'] == '22 TO 24', 'storey'] = '4-higher'
    df.loc[df['storey_range'] == '25 TO 27', 'storey'] = '4-higher'
    df.loc[df['storey_range'] == '28 TO 30', 'storey'] = '4-higher'
    df.loc[df['storey_range'] == '31 TO 33', 'storey'] = '5-skyscraper'
    df.loc[df['storey_range'] == '34 TO 36', 'storey'] = '5-skyscraper'
    df.loc[df['storey_range'] == '37 TO 39', 'storey'] = '5-skyscraper'
    df.loc[df['storey_range'] == '40 TO 42', 'storey'] = '5-skyscraper'
    df.loc[df['storey_range'] == '43 TO 45', 'storey'] = '5-skyscraper'
    df.loc[df['storey_range'] == '46 TO 48', 'storey'] = '5-skyscraper'
    df.loc[df['storey_range'] == '49 TO 51', 'storey'] = '5-skyscraper'
    return df

def make_datetime(df):
    df['month'] = pd.to_datetime(df['month'])
    return df

def create_features(df):
    df['yr'] = df['month'].dt.year
    df['mth'] = df['month'].dt.month
    df['qtr'] = df['month'].dt.quarter
    df['age'] = df['yr'] - df['lease_commence_date']
    return df

class Datapipeline():
    def __init__(self):
        # initialise model.
        self.numeric_features = ['floor_area_sqm', 'lease_commence_date', 'age']
        self.nominal_features = ['town', 'flat_type', 'street_name', 'flat_model']
        self.ordinal_features = ['storey', 'yr', 'mth', 'qtr']
        # self.target = ['resale_price']

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        ordinal_transformer = Pipeline(steps=[('scaler', OrdinalEncoder())])

        self.preprocessor = ColumnTransformer(
            transformers = [('num', numeric_transformer, self.numeric_features),
                            ('nom', nominal_transformer, self.nominal_features),
                            ('ord', ordinal_transformer, self.ordinal_features)])

    def process_data(self):
        df = fetch_data(r'data/resale-flat-prices-based-on-registration-date-from-2016-to-2017.csv')
        df = df.drop_duplicates(keep='last')
        df = regroup_storey(df)
        df = make_datetime(df)
        df = create_features(df)
        cols_to_keep = ['town', 'flat_type', 'street_name', 'floor_area_sqm', 'flat_model', 
                        'lease_commence_date', 'resale_price', 'storey', 'yr', 'mth', 'qtr', 'age']
        df = df[cols_to_keep]
        return df

    def get_train_data_regression(self):
        """
        Description of the function. 

        :param train_data_path: ......
        :return: ......
        """
        df = self.process_data()
        X = df.drop('resale_price', axis = 1)
        y = df['resale_price'] / 1000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        self.preprocessor.fit(X_train)
        transformed_columns = (self.numeric_features + 
                            list(self.preprocessor.named_transformers_['nom'].named_steps['onehot'].get_feature_names(self.nominal_features)) +
                            self.ordinal_features)
        X_train = pd.DataFrame(self.preprocessor.transform(X_train).todense(), columns = transformed_columns)
        return X_train, y_train  

    def get_test_data_regression(self):
        """
        Description of the function. 

        :param train_data_path: ......
        :return: ......
        """
        df = self.process_data()
        X = df.drop('resale_price', axis = 1)
        y = df['resale_price'] / 1000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        self.preprocessor.fit(X_train)
        transformed_columns = (self.numeric_features + 
                            list(self.preprocessor.named_transformers_['nom'].named_steps['onehot'].get_feature_names(self.nominal_features)) +
                            self.ordinal_features)

        X_test = pd.DataFrame(self.preprocessor.transform(X_test).todense(), columns = transformed_columns)
        return X_test, y_test

    def get_train_data_classification(self):
        """
        Description of the function. 

        :param train_data_path: ......
        :return: ......
        """
        df = self.process_data()
        X = df.drop('resale_price', axis = 1)
        y = df['resale_price'] / 1000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        self.preprocessor.fit(X_train)
        transformed_columns = (self.numeric_features + 
                            list(self.preprocessor.named_transformers_['nom'].named_steps['onehot'].get_feature_names(self.nominal_features)) +
                            self.ordinal_features)
        X_train = pd.DataFrame(self.preprocessor.transform(X_train).todense(), columns = transformed_columns)

        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        
        y_train = pd.DataFrame(y_train)
        discretizer.fit(y_train)
        y_train = discretizer.transform(y_train)
        y_train = y_train.ravel()
        return X_train, y_train  

    def get_test_data_classification(self):
        """
        Description of the function. 

        :param train_data_path: ......
        :return: ......
        """
        df = self.process_data()
        X = df.drop('resale_price', axis = 1)
        y = df['resale_price'] / 1000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        self.preprocessor.fit(X_train)
        transformed_columns = (self.numeric_features + 
                            list(self.preprocessor.named_transformers_['nom'].named_steps['onehot'].get_feature_names(self.nominal_features)) +
                            self.ordinal_features)
        X_test = pd.DataFrame(self.preprocessor.transform(X_test).todense(), columns = transformed_columns)

        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_train = pd.DataFrame(y_train)
        discretizer.fit(y_train)
        y_test = pd.DataFrame(y_test)
        y_test = discretizer.transform(y_test)
        y_test = y_test.ravel()
        return X_test, y_test