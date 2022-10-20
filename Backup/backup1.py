# ECSE 551 - A1
# %---------------------------------------------- Import Classes ---------------------------------------------------% #
import sys
import numpy                as np
from Classes import Dataset as Dset
from Classes import Model   as Mdl
from Classes import Test    as Tst
# %--------------------------------------------------- Main --------------------------------------------------------% #
def main():
    # [Running Parameters for Debug]
    writeTofile = True
    filePath    = "./OutPut.txt"
    
    # [Parameteres] : Data Set
    visualizeData = False
    pltStd, pltMean, pltSpread, pltHist = False, False, False, False

    # [Parameteres] : Data Files to read from
    dataFiles = ["./air_quality.csv",
                 "./liver_patient.csv"]
    
    # [Parameteres] : Model Accuracy
    tols      = [1e-3, 1e-4, 1e-5]
    max_iters = [1e+2, 1e+3, 1e+4]
    numFolds  = 10
    
    # [Parameteres] : Methods
    methods = [["Gradient Decent",                 "GD", "LogReg"]]
            #    ["Linear Discriminant Analysis",    "DA", "Linear"],
            #    ["Quadratic Discriminant Analysis", "DA", "Quadratic"]]
    
    # [Parameteres] : HyperParameters
    alphas = ["lambda k : 1/(5*(k+1))",
              "lambda k : 1/(10*(k+1))",
              "lambda k : 1/(50*(k+1))",
              "lambda k : 1/(100*(k+1))",
              "lambda k : 1/((k+1)**2)",
              "lambda k : 0.01",
              "lambda k : 0.005",
              "lambda k : 0.0025",
              "lambda k : 0.001",
              "lambda k : 0.0005",
              "lambda k : 0.00025"]

    lasos  = [0, 0.05, 0.1]
    
    # [Parameteres] : Remove Identical Observations
    rmvSimilarRows = True
       
    # %----------------------------------------- Air Quality Testing -----------------------------------------------% #
    tDataSet = Dset.Dataset(dataFiles[0])
    if (visualizeData): 
        tDataSet.analyzeData(pltStd, pltMean, pltSpread, pltHist)
    
    test = Tst.Test(tDataSet, methods, numFolds,
                    alphas, lasos,
                    max_iters, tols,
                    rmvSimilarRows,
                    writeTofile, filePath)
    
    # %-------------------------------------------- Unit Tests ----------------------------------------------------% #
    # Purpose: Impact of removing identical observations
    test.rmvRowImpact()
    '''
        Result: 
        - Removing identical rows:
        - There were 240 identical observations with identical classes
        - There were 0   identical observations within both classes
        - - Reduces Error from 25.58 -> 25.46
        Conclusion:
        - Will decide to remove rows for the rest of the test
        - - In doing this removing/adding features have no effect!!??
        - - Best Performance 25.85
        - Test with not removing
        - - Best Performance 25.72
    '''
    # ! Conclusion Result:
    test.rmvSimilarRows = False
    
    # Purpose: Impact of Removing each Feature
    test.rmvFeature()
    '''
        Result: 
        - For Grad Decent: 
        - - Removing feature 4 improved accuracy
        - - Removing feature 9 had no impact
        - - Removing features 1 and 5 had a slight negative impact
        - For LDA:
        - For QDA:
    '''
    
    # Purpose: Impact of Removing a subset of Features
    combins = [[1, 9], [1, 4], [4, 9], [1, 4, 9],
                [5, 9], [5, 4], [1, 5], [5, 4, 9],
                [1, 5, 4, 9], [3, 5], [3, 8], [5, 8], [3, 5, 8]]
    test.rmvCombFeatures(combins)
    '''
        Result: 
        - For Grad Decent: 
        - - Removing feature [9, 4] 
        - - - no impact on K-Fold error            [25.72]
        - - - biggest improvement on overall error [25.28 -> 25.20]
        - For LDA:
        - For QDA:
        - Across all, removing features [9, 4] improved accuracy
        - - However, Removing just 4 yeilds better gain
        Conclusion:
        - Will remove features 4 and 9 from now on
    '''
    # ! Conclusion Result:
    rmvedFeatures = np.array([4, 9]) - 1
    test.dataSet = Dset.Dataset(np.delete(test.dataSet.data, rmvedFeatures, axis=1),
                                test.dataSet.name)
    
    # Purpose: Impact of Adding Feature Complexity
    flist = ["lambda x: np.log(x + 1)",
             "lambda x: np.log10(x + 1)",
             "lambda x: np.sqrt(x)",
             "lambda x: x**2",
             "lambda x: x**3",
             "lambda x: x**4"]
    test.addFeatureComplexity(flist)
    '''
        Result: 
        - For Grad Decent: 
        - - log(f) -> worse
        - - log10(f6 + 1): 25.20 -> 25.08
        - - f5^2:          25.20 -> 25.08
        
        
    '''
    
    # Purpose: Impact of Adding a Set of Feature Complexities
    flist = [[[1, "lambda x: x**3"],            [7, "lambda x: np.log10(x + 1)"]],
             [[1, "lambda x: x**3"],            [2, "lambda x: np.log(x + 1)"]],
             [[1, "lambda x: x**3"],            [5, "lambda x: np.sqrt(x)"]],
             [[7, "lambda x: np.log10(x + 1)"], [2, "lambda x: np.log(x + 1)"]],
             [[7, "lambda x: np.log10(x + 1)"], [5, "lambda x: np.sqrt(x)"]],
             [[2, "lambda x: np.log(x + 1)"],   [5, "lambda x: np.sqrt(x)"]],
             [[2, "lambda x: np.log(x + 1)"],   [5, "lambda x: np.sqrt(x)"], [1, "lambda x: x**3"]]]   
    # test.addFeatureComplexitySet(flist)  
    '''
        Result: 
        - For Grad Decent: 
        - - On Average, the case:
        - - [[1, "lambda x: np.log(x + 1)"], [4, "lambda x: np.sqrt(x)"], [0, "lambda x: x**3"]]
        - - Performed best
    '''
    # ! Conclusion Result:
    addList = [[2, "lambda x: np.log(x + 1)"], [5, "lambda x: np.sqrt(x)"], [1, "lambda x: x**3"]]  
    X = np.copy(test.dataSet.data[:,:-1])
    for f in addList:
        X = Mdl.Model.addModFeature(X, f[0]-1, eval(f[1]))
    test.dataSet = Dset.Dataset(np.c_[X,test.dataSet.data[:,-1]],
                                test.dataSet.name)
    
    # Purpose: Finding Optimal Stepsize and Laso const for GD
    # test.findOptimalAlphaLambda()
    '''
        Result: 
        - For Grad Decent: 
        - - On Average, the case:
        - - lambda k : 1/(10*(k+1)) Performed best
        - - - Across all cases, the lasso reg did not help
    '''
    # ! Conclusion Result:
    test.alphas = ["lambda k : 1/(10*(k+1))"]
    test.lasos  = [0]
    
    # Purpose: Finding Optimal tol and max_iter number
    # test.findOptimalTolMaxIter()
    
if __name__ == '__main__':
    np.random.seed(69)
    print("Start")
    main()
    sys.stdout.close()























