# Column Names
BEHAVIOR = 'BEHAVIOR'
ACCELERATION_X = 'ACCX'
ACCELERATION_Y = 'ACCY'
ACCELERATION_Z = 'ACCZ'
NEW_INDEX = 'INPUT_INDEX'
COLUMN_NAMES = [ BEHAVIOR,
            ACCELERATION_X, 
            ACCELERATION_Y, 
            ACCELERATION_Z ]

WARNING_COLUMN_MISSING = " column is missing"

# Behavior that has no x,y,z
NO_VIDEO = 'n'

# Configuration
WINDOW_SIZE = 25
CLASSES_OF_INTEREST = "hlmstw"
CLASSES_OF_INTEREST_LIST = ['l','s','t','w']
CLASS_INDICES = {
    'h' : 0,
    'l' : 1,
    'm':  2,
    's' : 3,
    't' : 4,
    'w' : 5,
}
STRIKES = ['h', 'm']

# Models
RANDOM_FOREST = "Random_Forest"
SVM = "SVM"
NAIVE_BAYES = "Naive_Bayes"

MODEL_NAMES = {
    "rf": RANDOM_FOREST,
    "svm" : SVM,
    "nb" : NAIVE_BAYES,
}

# Types of Sampling
NO_SAMPLING = "No_Sampling"
SMOTE = "SMOTE"
ADASYN = "ADASYN"
RANDOM_OVERSAMPLE = "Random_Over_Sampler"

SAMPLING_TECHNIQUE_NAMES = {
    "o": RANDOM_OVERSAMPLE,
    "s" : SMOTE,
    "a" : ADASYN,
    'ns' : NO_SAMPLING,
}
