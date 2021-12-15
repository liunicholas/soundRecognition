
NEW_MODEL = True              #tests on a new model

#set both to true for purely predicitng date with specified model
PREDICT_ON_DATE = False        #set to true to predict day
OVERRIDE = False               #overrides load, train, test, and new_model

#vars for predicting single date
predictDate = "2021-05-20"
savedModelName = "10_10_remoteVersionTesting"

graphPath = "./info/pyplots/newestPlot.png"                  #save mpl graph
dataPath = "./info/datasets/allSpy.npy"                      #save npy arrays
checkpointPath = "./info/checkpoints"                        #save models
stocksIncludedPath = "./info/datasets/stocksIncluded.txt"    #save list of stocks used

savedModelsPath = "./savedModels"                            #save best model
previousSavePath = f"{savedModelsPath}/{savedModelName}"    #location of desired model for predicting
