# ECSE 551 - A1
# Authors: 
# - Louis Lortie
# - Sepehr Moalemi
# - Syed Shabbir Ahmed
# %---------------------------------------------- Import Classes ---------------------------------------------------% #
import sys
import numpy                as np
from Classes import Dataset as Dset
from Classes import Model   as Mdl
from Classes import Test    as Tst
# %--------------------------------------------- Global Parameters -------------------------------------------% #
# [Running Parameters] : Control output of tests
verbos      = True
writeTofile = True
OutputPath1 = "./Results/Air_Quality_Results.txt"
OutputPath2 = "./Results/Liver_Patient_Results.txt"

# [Running Parameteres] : Various Plots Analyzing the data set
visualizeData = False
pltStd, pltMean, pltSpread, pltHist = True, True, True, True

# [Parameteres] : Data Files to read from
dataFiles = ["./Data/air_quality.csv",
             "./Data/liver_patient.csv"]

# [Parameteres] : Model Accuracy
tols      = [1e-3, 1e-4, 1e-5]
max_iters = [1e+4, 1e+5, 1e+6]

# [Parameters] : K-Fold Cross Validation
numFolds  = 10

# [Parameteres] : Methods
methods = [["Gradient Decent",                 "GD", "LogReg"],
           ["Linear Discriminant Analysis",    "DA", "Linear"],
           ["Quadratic Discriminant Analysis", "DA", "Quadratic"]]

# ! Note: Uppon testing, GD performs better than both DA
# ! So for the rest of the code, run tests on GD    
methods = [["Gradient Decent", "GD", "LogReg"]]

# [Parameteres] : HyperParameters
alphas = ["lambda k : 1/(5*(k+1))",
          "lambda k : 1/(10*(k+1))",
          "lambda k : 1/(50*(k+1))",
          "lambda k : 1/((k+1)**2)",
          "lambda k : 0.01",
          "lambda k : 0.001",
          "lambda k : 0.0001",
          "lambda k : 0.00001"]

lasos  = [0, 0.05, 0.1, 0.5]

# [Parameteres] : Remove Identical Observations
rmvSimilarRows = True

# %--------------------------------------------------- Main --------------------------------------------------% #
def main():
    getOptimalModel(dataFiles[0], verbos)
    getOptimalModel(dataFiles[1], verbos)
       
# %----------------------------------------- Air Quality Testing ---------------------------------------------% #
def getOptimalModel(fileName, verbos=False):
    if("air_quality.csv" in fileName):
        tDataSet = Dset.Dataset(fileName)
        if (visualizeData): 
            tDataSet.analyzeData(pltStd, pltMean, pltSpread, pltHist)
        
        test = Tst.Test(tDataSet, methods, numFolds,
                        alphas, lasos,
                        max_iters, tols,
                        rmvSimilarRows,
                        writeTofile, OutputPath1)
    
    # %-------------------------------------------- Unit Tests ----------------------------------------------% #
        # Purpose: Impact of removing identical observations
        if(verbos): test.rmvRowImpact()
        '''
            Result: 
            - Removing identical rows:
            - There were 240 identical observations with identical classes
            - There were 0   identical observations within both classes
            - - Reduces Error from 25.58 -> 25.46
            Conclusion:
            - Will decide to Not to remove rows for the rest of the test
        '''
        # ! Conclusion Result:
        test.rmvSimilarRows = False
        
        # Purpose: Finding Optimal Stepsize and Laso const for GD
        if(verbos): test.findOptimalAlphaLambda()
        '''
            Result: 
            - For Grad Decent: 
            - - On Average, the case:
            - - lambda k : 1/(5*(k+1)) Performed best
            - - - Use laso of 0.1
        '''
        # ! Conclusion Result:
        test.alphas = ["lambda k : 1/(5*(k+1))"]
        test.lasos  = [0.1]
        
        # Purpose: Finding Optimal tol and max_iter number
        if(verbos): test.findOptimalTolMaxIter()
                    
        # Purpose: Impact of Removing each Feature
        if(verbos): test.rmvFeature()
        '''
            Result: 
            - For Grad Decent: 
            - - Removing feature 4, 5 improved accuracy
            - - Removing feature 9 had no impact
        '''
        
        # Purpose: Impact of Removing a subset of Features
        combins = [[1, 9], [1, 4], [4, 9], [1, 4, 9],
                    [5, 9], [5, 4], [1, 5], [5, 4, 9],
                    [1, 5, 4, 9], [3, 5], [3, 8], [5, 8], [3, 5, 8]]
        if(verbos): test.rmvCombFeatures(combins)
        '''
            Result: 
            - For Grad Decent: 
            - - Removing feature [9, 4, 5] improved accuracy
            Conclusion:
            - Will remove features 4, 5, and 9 from now on
        '''
        # ! Conclusion Result:
        rmvedFeatures = np.array([5, 9]) - 1
        test.dataSet = Dset.Dataset(np.delete(test.dataSet.data, rmvedFeatures, axis=1),
                                    test.dataSet.name)
        
        # Purpose: Impact of Adding Feature Complexity
        flist = ["lambda x: np.log(x + 1)",
                "lambda x: np.log10(x + 1)",
                "lambda x: np.sqrt(x)",
                "lambda x: x**2",
                "lambda x: x**3",
                "lambda x: x**4"]
        if(verbos): test.addFeatureComplexity(flist)
        '''
            Result: 
            - For Grad Decent: 
            - f2**3, np.sqrt(f7), f3**3, improved
        '''
        
        # Purpose: Impact of Adding a Set of Feature Complexities  
        flist = [
                    [[2, "lambda x: x**3"],  [8, "lambda x: np.sqrt(x)"]],
                    [[3, "lambda x: x**3"],  [2, "lambda x: x**3"], [7, "lambda x: np.log(x + 1)"]],
                    [[2, "lambda x: x**3"],  [3, "lambda x: x**3"]],
                ] 
        if(verbos): test.addFeatureComplexitySet(flist)  
        '''
            Result: 
            - For Grad Decent: 
            - - On Average, the case:
            - - [2, "lambda x: x**3"],  [8, "lambda x: np.sqrt(x)"]
            - - Performed best
        '''
        # ! Conclusion Result:
        addList = [[2, "lambda x: x**3"],  [8, "lambda x: np.sqrt(x)"]]
        X = np.copy(test.dataSet.data[:,:-1])
        for f in addList:
            X = Mdl.Model.addModFeature(X, f[0]-1, eval(f[1]))
        test.dataSet = Dset.Dataset(np.c_[X,test.dataSet.data[:,-1]],
                                    test.dataSet.name)

        # Purpose: Run the Model on the full Dataset
        test.findErrOnAll(tDataSet)

# %----------------------------------------- Liver Patient Testing ---------------------------------------------% #
    if("liver_patient.csv" in fileName):
        tDataSet = Dset.Dataset(fileName)
        if (visualizeData): 
            tDataSet.analyzeData(pltStd, pltMean, pltSpread, pltHist)
        
        test = Tst.Test(tDataSet, methods, numFolds,
                        alphas, lasos,
                        max_iters, tols,
                        rmvSimilarRows,
                        writeTofile, OutputPath2)
    
    # %-------------------------------------------- Unit Tests ----------------------------------------------% #
        # Purpose: Impact of removing identical observations
        if(verbos): test.rmvRowImpact()
        '''
            Result:
            - Nothing to remove 

            Conclusion:
            - Will not remove
        '''
        # ! Conclusion Result:
        test.rmvSimilarRows = False

        # Purpose: Finding Optimal Stepsize and Laso const for GD
        if(verbos): test.findOptimalAlphaLambda()

        # ! Conclusion Result:
        test.alphas = ["lambda k : 1/(5*(k+1))"]
        test.lasos  = [0.0]
        
        # Purpose: Impact of Removing each Feature
        if(verbos): test.rmvFeature()
        '''
            Result: 
            - For Grad Decent: 
            - - No modification
            - - No need to remove a set of features
            Conclusion:
            - No modifications
        '''
        
        # Purpose: Finding Optimal tol and max_iter number
        if(verbos): test.findOptimalTolMaxIter()
        
        # Purpose: Impact of Adding Feature Complexity
        flist = ["lambda x: np.log(x + 1)",
                "lambda x: np.log10(x + 1)",
                "lambda x: np.sqrt(x)",
                "lambda x: x**2",
                "lambda x: x**3",
                "lambda x: x**4"]
        if(verbos): test.addFeatureComplexity(flist)
        '''
            Result: 
            - For Grad Decent: 
            - The below combinations resulted in noticable improvements 
            - [Feature], [Function]
            - [8, "lambda x: np.log(x + 1)"], 
            - [6, "lambda x: x**2"], 
            - [3, "lambda x: np.sqrt(x)"], 
            - [4, "lambda x: x**2"], 
            - [5, "lambda x: np.sqrt(x)"]
        '''
        
        # Purpose: Impact of Adding a Set of Feature Complexities  
        # flist = [ 
        #             [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [3, "lambda x: np.sqrt(x)"]],
        #             [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [5, "lambda x: np.sqrt(x)"]],
        #             [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [4, "lambda x: x**2"]],
                    
        #             [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [3, "lambda x: np.sqrt(x)"], [5, "lambda x: np.sqrt(x)"]],
        #             [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [3, "lambda x: np.sqrt(x)"], [4, "lambda x: x**2"]],
                    
        #             [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [3, "lambda x: np.sqrt(x)"], [4, "lambda x: x**2"], [5, "lambda x: np.sqrt(x)"]],
        #         ]
        
        flist = [[[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [4, "lambda x: x**2"]]]
        if(verbos): test.addFeatureComplexitySet(flist)  
        '''
            Result: 
            - For Grad Decent: 
            - - On Average, the case:
            - - [8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [4, "lambda x: x**2"]
            - - Performed best
        '''
        # ! Conclusion Result:
        addList = [[8, "lambda x: np.log(x + 1)"], [6, "lambda x: x**2"], [4, "lambda x: x**2"]]
        X = np.copy(test.dataSet.data[:,:-1])
        for f in addList:
            X = Mdl.Model.addModFeature(X, f[0]-1, eval(f[1]))
        test.dataSet = Dset.Dataset(np.c_[X,test.dataSet.data[:,-1]],
                                    test.dataSet.name)

        # Purpose: Run the Model on the full Dataset
        test.findErrOnAll(tDataSet)



# %------ Run ----------% #
if __name__ == '__main__':
    np.random.seed(1234)
    main()
    sys.stdout.close()