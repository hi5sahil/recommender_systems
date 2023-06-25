# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:25:39 2018

@author: Frank
"""


from MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from surprise import KNNBasic
from surprise import SVD, SVDpp
from Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# content based reco
contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# User-based KNN (variants : KNNWithZScore, KNNWithMeans, KNNBaseline)
# sim_options for name = cosine, msd, pearson
UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, "User KNN")

# Item-based KNN
# sim_options for name = cosine, msd, pearson
ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, "Item KNN")

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# doTopN = True
#evaluator.Evaluate(False)
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml, testSubject=410)


