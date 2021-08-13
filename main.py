from utils.Predictor import Predictor
from utils.dataProcessor import DataProcessor

dataProcessor = DataProcessor()
dataProcessor.plot_correlation()

predictor = Predictor()
predictor.trainModel(dataProcessor.preprocess())