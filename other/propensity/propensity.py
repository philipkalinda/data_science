from typing import List
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import logging
import s3fs

_log = logging.getLogger(__name__)


class PropensityModel(object):
    """
    Model used to predict likelihood that customers will buy a particular product.
    """
    
    def __init__(self,
                 target_variables: List[str],
                 feature_categories: List[str],
                 model=None,
                 normalise_features: bool = True):
        """
        Model used to predict likelihood that customers will buy a particular product.
        
        Keyword arguments:
        target_variables: list of product names for which propensity should be predicted
        feature_categories: subset of ['product_name','product_category','reporting_channel','recency','frequency','total_spend']
        model: a scikit learn classifier object (default: logistic regression)
        normalise_features: whether continuous features short be normalised to have (positive) unit standard deviation (default: True)
        """
        
        self.target_variables = target_variables
        self.feature_categories = feature_categories
        if model is None:
            self.model = OneVsRestClassifier(LogisticRegression(random_state=0, class_weight='balanced', solver='liblinear'))
        self.normalise_features = normalise_features
    
    def get_features_and_target(self, trades_features: pd.DataFrame, trades_target: pd.DataFrame) -> pd.DataFrame:
        """
        Construct dataframe where rows represent customers and columns represent features and target variable.
        
        Keyword arguments:
        trades_features: product purchases before a cut-off date used to calculate feature columns
        trades_target: product purchases after a cuf-off date used to calculate target column
        """
        
        sf_groups = trades_features.drop_duplicates(subset=['sf_account_id', 'trade_date', 'sku']).groupby('sf_account_id')

        # calculate features
        feature_dfs = []
        if 'product_name' in self.feature_categories:
            feature_dfs += [sf_groups.product_name.value_counts().unstack().notnull()]
        if 'product_category' in self.feature_categories:
            feature_dfs += [sf_groups.product_category.value_counts().unstack().notnull()]
        if 'reporting_channel' in self.feature_categories:
            feature_dfs += [sf_groups.sub_reporting_channel.value_counts().unstack().notnull()]
        if 'recency' in self.feature_categories:
            feature_dfs += [(trades_features.trade_date_dt.max()-sf_groups.trade_date_dt.max()).dt.days.to_frame().rename(columns={'trade_date_dt':'recency'})]
        if 'frequency' in self.feature_categories:
            feature_dfs += [sf_groups.product_name.count().to_frame().rename(columns={'product_name':'frequency'})]
        if 'total_spend' in self.feature_categories:
            feature_dfs += [sf_groups.cost_float.sum().to_frame().rename(columns={'cost_float':'total_spend'})]

        # concat features
        customer_df = pd.concat(feature_dfs, axis=1, sort=False)  # outer join on index

        # add target variable
        for target_variable in self.target_variables:
            if (trades_target.product_name == target_variable).any():
                customer_df['target_'+target_variable] = trades_target.groupby(['sf_account_id', 'product_name']).trade_date.any().unstack()[target_variable]
            else:
                customer_df['target_'+target_variable] = False

        # remove customers with no purchases before cut off
        customer_df = customer_df[customer_df[customer_df.columns[customer_df.columns != 'target']].any(axis=1)]

        # replace nans with False
        customer_df.fillna(False, inplace=True)

        return customer_df
    
    def get_train_and_test_sets(self,
                                all_trades_df: pd.DataFrame,
                                train_start: pd.Timestamp,
                                train_end: pd.Timestamp,
                                test_end: pd.Timestamp) -> pd.DataFrame:
        """
        Split data set of all product purchases into training and test sets based on start and end dates.
        Training set is further divided such that features are calculated from purchases before a cut-off,
        while the target is calculate from sales after that cut-off.
        
        Keyword arguments:
        all_trades_df: pandas dataframe where rows represent product purchases
        train_start: pandas timestamp representing start date of training set
        train_end: pandas timestamp representing end date of training set
        test_end: pandas timestamp representing end date of test set
        """
        
        # training set
        feature_and_target_cutoff = train_end - pd.Timedelta('120D') # last 4 months of training set
        trades_features = all_trades_df[(all_trades_df['trade_date_dt'] >= train_start) & (all_trades_df['trade_date_dt'] < feature_and_target_cutoff)]
        trades_target = all_trades_df[(all_trades_df['trade_date_dt'] >= feature_and_target_cutoff) & (all_trades_df['trade_date_dt'] < train_end)]
        training_set = self.get_features_and_target(trades_features, trades_target)
        training_set['train_or_test'] = 'train'

        # test set
        trades_features = all_trades_df[(all_trades_df['trade_date_dt'] >= train_start) & (all_trades_df['trade_date_dt'] < train_end)]
        trades_target = all_trades_df[(all_trades_df['trade_date_dt'] >= train_end) & (all_trades_df['trade_date_dt'] < test_end)]
        test_set = self.get_features_and_target(trades_features, trades_target)
        test_set['train_or_test'] = 'test'
        
        customer_df = pd.concat([training_set, test_set], sort=False)
        customer_df.fillna(False, inplace=True)
        customer_df.index.name = 'sf_account_id'
        
        return customer_df

    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train model with dataframe, returns None.
        Keyword arguments:
        train_df: pandas dataframe with rows representing customers and only feature and target columns + train_or_test indicator
        """

        # get feature list
        target_columns, features = PropensityModel.get_feature_and_target_columns(train_df)

        # train model
        x_train = train_df[features]
        y_train = train_df[target_columns]
        self.model.fit(x_train, y_train)

    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return predicted propensities for both training and test sets
        Keyword arguments:
        train_df: pandas dataframe with rows representing customers and only feature and target columns + train_or_test indicator
        test_df: pandas dataframe with rows representing customers and only feature and target columns + train_or_test indicator
        """

        # get feature list
        target_columns, features = PropensityModel.get_feature_and_target_columns(train_df)

        # predict propensities
        model_prediction_df_list = []
        for pred_df in [train_df, test_df]:
            x_test = pred_df[features]
            y_test = pred_df[target_columns]
            y_pred = self.model.predict_proba(x_test)
            
            # select second column (positive class) if there is only one target variable
            if len(self.target_variables) == 1:
                y_pred = y_pred[:,1]
                
            fold_predictions = pd.DataFrame(y_pred, columns=['prediction_'+x for x in self.target_variables])
            fold_predictions['sf_account_id'] = pred_df.index
            for column in target_columns:
                fold_predictions[column.replace('prediction_','target_')] = y_test[column].tolist()
            fold_predictions['train_or_test'] = pred_df.train_or_test.iloc[0]
            model_prediction_df_list += [fold_predictions]
        model_predictions = pd.concat(model_prediction_df_list, sort=False)

        return model_predictions
    
    @staticmethod
    def get_feature_and_target_columns(train_df):
        
        target_columns = train_df.columns[train_df.columns.str.contains('target')].tolist()
        features = train_df.columns[train_df.columns.isin(target_columns+['train_or_test', 'fold']) == False]
        
        return target_columns, features
    
    @staticmethod
    def clean_trades_df(all_trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data set of product purchases by dropping missing values, colours from cameras, tiers from subscriptions, Hive Live spikes.
        Also add prefixes to product categories and sales channels, and convert dates to pandas timestamps.
        
        Keyword arguments:
        all_trades_df: pandas dataframe with rows representing sales
        """
        
        trades_df = all_trades_df.copy()
        
        # drop missing values from important columns
        trades_df.dropna(subset=['sf_account_id', 'sub_reporting_channel', 'product_name', 'product_category', 'trade_date'], inplace=True)
        
        # remove colours from indoor camera names
        trades_df.replace(to_replace=['Hive View Camera (Black)', 'Hive View Camera (White)'], value='Hive View Camera', inplace=True)
        
        # remove tiers from camera subscriptions
        trades_df.replace(to_replace=['Video Playback v2 - Tier 1', 'Video Playback v2 - Tier 2', 'Video Playback v2 - Tier 3'], value='Video Playback', inplace=True)
        
        # remove large amount of hive lives that were all "sold" on the same two days
        trades_df = trades_df[(all_trades_df.product_name == 'Hive Live') & (all_trades_df.trade_date.isin(['2019-05-30', '2019-05-31'])) == False]
        
        # prefix character to allow features to be easily distinguished
        # trades_df['product_name'] = '+' + trades_df['product_name']
        trades_df['product_category'] = '_' + trades_df['product_category']
        trades_df['sub_reporting_channel'] = '-' + trades_df['sub_reporting_channel']
        
        # convert dates
        trades_df['trade_date_dt'] = pd.to_datetime(trades_df.trade_date)
        
        return trades_df
    
    @staticmethod
    def normalise_series(to_normalise: pd.Series) -> pd.Series:
        """
        Divide by standard deviation but don't subtract mean (so that features remain positive).
        
        Keyword arguments:
        to_normalise: pandas series representing individual feature
        """
        
        # return (to_normalise - to_normalise.mean()) / to_normalise.std() # 0 mean and unit standard deviation
        return to_normalise / to_normalise.std() # positive and unit standard deviation
    
    @staticmethod
    def clean_customer_df(customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean customer dataframe by removing customers with large number of purchases.
        
        Keyword arguments:
        customer_df: pandas dataframe where rows represent purchases
        """
        
        # remove customers with more than 20 purchases
        if 'frequency' in customer_df.columns:
            customer_df = customer_df[customer_df.frequency < 20]
        
        return customer_df
    
    @staticmethod
    def normalise_features(customer_df, features_to_normalise: List[str] = ['recency', 'frequency', 'total_spend']):
        """
        Normalise a list of (continuous) features using normalise_series()
        
        Keyword arguments:
        customer_df: pandas dataframe where rows represent purchases
        features_to_normalise: list of continuous features which should be normalised
        """
        
        for feature in features_to_normalise:
            if feature in customer_df.columns:
                customer_df[feature] = PropensityModel.normalise_series(customer_df[feature])
        return customer_df
    
    def write_predictions_to_s3(self, fold_predictions: pd.DataFrame, output_path: str):
        """
        Write DataFrame of propensity scores to S3 for each Salesforce ID.
        
        Keyword arguments:
        fold_predictions: DataFrame of propensity scores (prediction columns begin with prediction_)
        output_path: S3 path to store predictions as CSV (without prediction_ in column names)
        """
        
        # prepare dataframe
        prediction_columns = fold_predictions.columns[['prediction_' == x[:11] for x in fold_predictions.columns]].tolist()
        fold_predictions = fold_predictions[fold_predictions.train_or_test == 'test'] # only save test set
        fold_predictions = fold_predictions[['sf_account_id'] + prediction_columns] # only save salesforce ID and prediction columns
        fold_predictions.columns = ['sf_account_id'] + [x[11:] for x in prediction_columns] # remove predicted_ from column names
        
        # write to S3

        now_timestamp = str(pd.Timestamp.now()).split(".")[0]
        output_object = f'{output_path}propensity_{now_timestamp}.csv'
        csv_string = fold_predictions.to_csv(index=False)

        if 's3' in output_path:
            fs = s3fs.S3FileSystem()
            with fs.open(output_object, 'wb') as f:
                f.write(csv_string.encode())
        else:
            with open(output_object, 'wb') as f:
                f.write(csv_string.encode())

        return output_object

    
    def run_full_pipeline(self,
                          trades_csv_path: str,
                          output_path: str,
                          train_start: pd.Timestamp = None,
                          train_end: pd.Timestamp = None,
                          test_end: pd.Timestamp = None) -> pd.DataFrame:
        """
        Load purchase data from S3 and return predicted propensity scores for target product.
        trades_csv_path: s3 or local file system path to trades data
        train_start: pandas timestamp representing start date of training set
        train_end: pandas timestamp representing end date of training set
        test_end: pandas timestamp representing end date of test set
        """
        
        _log.info('loading trades...')
        all_trades_df = pd.read_csv(trades_csv_path, usecols=['sf_account_id',
                                                              'product_name',
                                                              'trade_date',
                                                              'product_category',
                                                              'sub_reporting_channel',
                                                              'sku'])
        
        _log.info('cleaning trades...')
        all_trades_df = PropensityModel.clean_trades_df(all_trades_df)
        
        # set limits if not set automatically
        if train_start is None:
            train_start = all_trades_df.trade_date_dt.min()
        if train_end is None:
            train_end = all_trades_df.trade_date_dt.max()
        if test_end is None:
            test_end = train_end + pd.Timedelta('14D')
        
        _log.info('reshaping trades...')
        customer_df = self.get_train_and_test_sets(all_trades_df,
                                                   train_start,
                                                   train_end,
                                                   test_end)
        
        _log.info('cleaning customers...')
        customer_df = PropensityModel.clean_customer_df(customer_df)
        
        _log.info('normalising customers...')
        if self.normalise_features:
            customer_df = PropensityModel.normalise_features(customer_df)

        # separate into train and test data frames
        train_df = customer_df[customer_df.train_or_test == 'train']
        test_df = customer_df[customer_df.train_or_test == 'test']

        _log.info('training...')
        self.train(train_df)

        _log.info('predicting...')
        fold_predictions = self.predict(train_df, test_df)
        
        _log.info('write predictions to S3...')
        object_path = self.write_predictions_to_s3(fold_predictions, output_path)
           
        _log.info('done')
        
        return object_path
