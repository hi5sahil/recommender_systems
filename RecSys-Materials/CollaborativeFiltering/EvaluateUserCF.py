# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData
from Evaluator import Evaluator


def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

ml, data, rankings = LoadMovieLensData()

evalData = EvaluationData(data, rankings)

# Train on leave-One-Out train set
trainSet = evalData.GetLOOCVTrainSet()

# user-based
sim_options = {'name': 'cosine',
               'user_based': True
               }


model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

leftOutTestSet = evalData.GetLOOCVTestSet()

# Build up dict to lists of (int(movieID), predictedrating) pairs
topN = defaultdict(list)
k = 10
for uiid in range(trainSet.n_users):
    # Get top N similar users to this one
    similarityRow = simsMatrix[uiid]
    
    similarUsers = []
    for innerID, score in enumerate(similarityRow):
        if (innerID != uiid):
            similarUsers.append( (innerID, score) )
    
    kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
    
    # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
    candidates = defaultdict(float)
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        userSimilarityScore = similarUser[1]
        theirRatings = trainSet.ur[innerID]
        for rating in theirRatings:
            candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
        
    # Build a dictionary of stuff the user has already seen
    watched = {}
    for itemID, rating in trainSet.ur[uiid]:
        watched[itemID] = 1
        
    # Get top-rated items from similar users:
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            topN[int(trainSet.to_raw_uid(uiid))].append( (int(movieID), 0.0) )
            pos += 1
            if (pos > 40):
                break
    
# Measure
#print(topN)     
                
#print(leftOutTestSet.head())            
            
print("HR", RecommenderMetrics.HitRate(topN, leftOutTestSet))   
#print("cHR", RecommenderMetrics.CumulativeHitRate(topN, leftOutTestSet)) 
#print("rHR", RecommenderMetrics.RatingHitRate(topN, leftOutTestSet)) 
#print("ARHR", RecommenderMetrics.AverageReciprocalHitRank(topN, leftOutTestSet)) 


# Print user coverage with a minimum predicted rating of 4.0:
#print("Coverage", RecommenderMetrics.UserCoverage(topN, data.GetFullTrainSet().n_users, ratingThreshold=4.0))
# Measure diversity of recommendations:
#print("Diversity", RecommenderMetrics.Diversity(topN, data.GetSimilarities()))
            
# Measure novelty (average popularity rank of recommendations):
#print("Novelty", RecommenderMetrics.Novelty(topN, data.GetPopularityRankings()))


'''
# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

evaluator.AddAlgorithm(model, "Item KNN")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
'''

