from utils.Predictor import Predictor
from utils.dataProcessor import DataProcessor

dataProcessor = DataProcessor()
dataProcessor.preprocess()

predictor = Predictor(dataProcessor.df)
predictor.predict()