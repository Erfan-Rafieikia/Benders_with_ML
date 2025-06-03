from data import Data

SEED = 42  # Ensuring reproducibility
data = Data # Data class for storing problem data


# Feature vector generation configuration
# --------- Problem Configuration ---------
N_F = 50                     # Number of first-stage decision trials
BINARY_FRACTION = 0.4        # Fraction of binary decisions among facilities

# --------- Random Walk Parameters ---------
N_WALK = 15                  # Number of random walks per node
L_WALK = 30                  # Length of each random walk

# --------- Word2Vec Parameters ---------
FEATURE_VECTOR_SIZE =10    # Size of the embedding vector for each scenario
WINDOW_SIZE = 5              # Context window size for Word2Vec
MIN_COUNT = 1                # Minimum count threshold for Word2Vec
SG = 1                       # Word2Vec model: 1 for Skip-gram, 0 for CBOW



#SFLC problem configuration
NUM_CUSTOMERS = 10 #Number of customers
NUM_FACILITIES = 5 # Number of facilities
DEMANDS = (1, 100)  # the range of customer demands
CAPACITIES = (500, 1000)  # the range of facility capacities
FIXED_COSTS = (2000, 5000)  # the range of facility fixed costs
SHIPMENT_COSTS = (1, 10)  # the range of shipment costs
NUM_SCENARIOS = 10 # Number of demand scenarios generated
VARIANCE_FACTOR = 0.2 # # Variance factor for scenario generation


#Representative scenario selection configuration
P_MEDIAN = 5  



#fcl PROBLEM 
DATA_FILE = "p1"


NUM_REPLICATIONS = 10 

#Making FLC problem stochastic by defining scenarios for demand 
NUM_SCENARIOS = 60 #Number of demand scenario generated
VARIANCE_FACTOR=0.2
P_MEDIAN = 5 #Number of sampled subproblems 

#Ml-augmented Benders structure
USE_PREDICTION = True # If true, use ML for creation of cuts on unsampled subproblems
SOLVE_UNSELECTED_IF_NO_CUTS = True  # If true, it will solve unsampled subproblems if other conditions like no ml generated cuts are met. 
PREDICTION_METHOD = "knn" # ML type uses for cut generation of unsampled subproblems
N_NEIGHBORS = 3 #How many nearest neighbor to consider if using KNN method. 

#Solving scenario for getting input for feature generation using random walk
N_F = 40  # Number of y_values_features
BINARY = False #Generation of y values to solve different subproblems for the final purpose of feature creation
#the forer was 20 


#Random Walk 
N_WALK = 10 #Number of random walks to geenrate from each node
L_WALK = 20 # Length of the random walk 
W = 5



