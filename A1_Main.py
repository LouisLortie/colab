# ECSE 551 - A1
# %---------------------------------------------- Import Classes ---------------------------------------------------% #
import inspect
import numpy as np
from Classes import Dataset as Dset
from Classes import Model   as Mdl
# %--------------------------------------------------- Main --------------------------------------------------------% #
def main():
    # [Parameteres] : Data Set
    visualizeData = True
    pltStd, pltMean, pltSpread, pltHist = False, False, False, True

    # [Parameteres] : Data Files to read from
    dataFiles = ["/home/louis/Documents/mcgill_classes/ecse_551/assignment1/colab/air_quality.csv",
                 "/home/louis/Documents/mcgill_classes/ecse_551/assignment1/colab/liver_patient.csv"]
    
    # [Parameteres] : Model Accuracy
    tol      = 1e-5
    max_iter = 1e3
    numFolds = 10
    
    # [Parameteres] : Methods
    methods = [["Least Squares",                   "LS"], 
               ["Gradient Decent",                 "GD", "LinReg"],
               ["Gradient Decent",                 "GD", "LogReg"],
               ["Linear Discriminant Analysis",    "DA", "Linear"],
               ["Quadratic Discriminant Analysis", "DA", "Quadratic"]]
    
    # [Parameteres] : HyperParameters
    alphas = [lambda k : 1/(10*(k+1)),
              lambda k : 1/((k+1))]
            
    # Generate Model for each file
    line = "-" * 50
    for file in dataFiles:
        # Generate Model from list of CSV Files
        tDataSet = Dset.Dataset(file)
        if (visualizeData): 
            Dset.Dataset.analyzeData(tDataSet,  pltStd, pltMean, pltSpread, pltHist)
            
        print(f'\nFinished Reading from [{file}] file.')

        # Generate Model
        tModel = Mdl.Model(tDataSet, [], lambda k : 0, numFolds, tol, max_iter)
        print(f'Generated Training Model for {tDataSet.name}.')

        # Extract Data
        X = tModel.X
        Y = tModel.Y
        
        # ! Testing removing features
        if (file == "./air_quality.csv"):
            X = Mdl.Model.rmFeaturesByIndx(tModel.X, [4, 9])
        
        # Train Model Using List of Methods
        for i, method in enumerate(methods):
            print(line*2)
            print(f'\n{i+1}) Run {method[0]} Model for {tDataSet.name}:')
            tModel.method = method
            
            # Train Model for different hyper parameters
            for j, alpha in enumerate(alphas):
                print(line)
                print(f'\n{i+1}.{j+1}) Run {method[0]} Model for {tDataSet.name}')
                alphastr = str(inspect.getsourcelines(alphas[j])[0][0])
                print(f'Using{alphastr}')
                
                # K-Fold Cross Validation
                tModel.alpha = alpha
                err = Mdl.Model.k_FoldCrossVal(tModel, X, Y, verbos=False)
                print(f'Accros {numFolds}-Fold Cross Validation\n')
                print(f'False Positive = {err[0]:.2f}%')
                print(f'False Negative = {err[1]:.2f}%')
                print(f'Error          = {100 - err[2]:.2f}%')
                print(f'Accuracy       = {err[2]:.2f}%')

                
                # Error Accross the full dataset
                Y_pred = Mdl.Model.fit_eval(tModel, X, X, Y)
                Mdl.Model.evalAccuracy(Y, Y_pred, verbos=False)
        
if __name__ == '__main__':
    print(f'\n------- Start --------')
    main()
    print(f'------ Finish --------')























