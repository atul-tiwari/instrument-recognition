import pandas as pd
import numpy as np

d1 = pd.read_csv("IRMAS-TrainingData/electric guitar.csv",index_col=0)
d2 = pd.read_csv("IRMAS-TrainingData/flute.csv",index_col=0)
d3 = pd.read_csv("IRMAS-TrainingData/piano.csv",index_col=0)

_Guitar = d1.values
_Flute = d2.values
_Paino = d3.values

#print(_Guitar.shape,_Flute.shape,_Paino.shape)

_ini_Data = np.concatenate((_Guitar,_Flute,_Paino),axis=0)

#print(_ini_Data.shape)

_Target  = np.array([1]*_Guitar.shape[0]+[2]*_Flute.shape[0]+[3]*_Paino.shape[0])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,mutual_info_classif,f_regression,mutual_info_regression


classes_fs = [f_classif,mutual_info_classif,f_regression,mutual_info_regression  ]



feature_selection_func =0
Features_req = 65

def _get_Data(Features=65,_class=0):

    feature_selection_func =_class
    Features_req = Features
    bestfeatures = SelectKBest(score_func=classes_fs[feature_selection_func], k=Features_req)
    fit = bestfeatures.fit(_ini_Data,_Target)

    dfscores = pd.DataFrame(fit.scores_)

    dfcolumns = pd.DataFrame(list(range(115)))

    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    featureScores.columns = ['Specs','Score'] 

    features = np.array(featureScores.nlargest(Features_req,'Score').values[:,0],dtype=int)

    return _ini_Data[:,features]

def _get_target():
    return _Target