import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.componenets.data_ingestion import DataIngestion
from src.componenets.data_tranformations import DataTransformation
from src.componenets.model_trainer import Model_Trainer

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path , test_data_path = obj.initiate_data_ingestion()
    DataTransformation = DataTransformation()
    train_arr , test_arr , _ = DataTransformation.initiate_data_tranformation(train_data_path,test_data_path)
    model_trainer = Model_Trainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
