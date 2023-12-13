
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import save_object
from src.Airbnb.exception import customexception
from sklearn.preprocessing import LabelEncoder,StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts','Preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            # cat and num columns
            cat_columns = train_df.columns[train_df.dtypes=="object"]
            num_columns = train_df.columns[train_df.dtypes!="object"]
            logging.info(f'Categorical Columns : {cat_columns}')
            logging.info(f'Numerical Columns : {num_columns}')

            logging.info("Data Pre-processing Initiated")

            train_df.last_review.fillna(method="ffill",inplace=True)
            test_df.last_review.fillna(method="ffill",inplace=True)

            train_df.first_review.fillna(method="ffill",inplace=True)
            test_df.first_review.fillna(method="ffill",inplace=True)

            train_df.host_since.fillna(method="ffill",inplace=True)
            test_df.host_since.fillna(method="ffill",inplace=True)
            # Handling missing values
            null_columns = ["bathrooms", "beds" ,"bedrooms"]
            for col in null_columns:
                train_df[col] = train_df[col].fillna((train_df[col].median()))
                test_df[col] = test_df[col].fillna((test_df[col].median()))
            logging.info("Null values imputed with median")

            # Handling Amenities Column in Training Data
            amenities_count_train = []
            for i in train_df["amenities"]:
                amenities_count_train.append(len(i))
            train_df["amenities"] = amenities_count_train

            # Handling Amenities Column in Testing Data
            amenities_count_test = []
            for i in test_df["amenities"]:
                amenities_count_test.append(len(i))
            test_df["amenities"] = amenities_count_test

            logging.info("Amenities column handled")

            train_df = train_df.dropna()
            test_df = test_df.dropna()
            
            target_column_name = 'log_price'
            drop_columns = [target_column_name,'id',"name","description","first_review","host_since","last_review","neighbourhood","thumbnail_url", "zipcode"]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            logging.info("Input and Target features created for Training Dataset")
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Input and Target features created for Testing Dataset")



            logging.info("Transforming the Input and Target Featres of Training and Testing Datasets")
            # Combine the training and testing datasets for fitting the LabelEncoder
            combined_df = pd.concat([input_feature_train_df, input_feature_test_df], axis=0)
            preprocessing_obj = LabelEncoder()
            req_col = ['property_type','room_type','bed_type','cancellation_policy','city','host_has_profile_pic','host_identity_verified','host_response_rate','instant_bookable']
            for col in req_col:
                combined_df[col] = preprocessing_obj.fit_transform(combined_df[col])
            input_feature_train_df[req_col] = combined_df[req_col][:len(input_feature_train_df)]
            input_feature_test_df[req_col] = combined_df[req_col][len(input_feature_train_df):]
            logging.info("Applying preprocessing object on training and testing datasets.")

            
            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]
            logging.info("Training and Testing data transformed into arrays")
            logging.info(f'Train Array : \n{train_arr}')
            logging.info(f'Test Array : \n{test_arr}')
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            logging.info("preprocessing pickle file saved")
            return(train_arr,test_arr)
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)