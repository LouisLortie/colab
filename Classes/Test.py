# ECSE 551 - A1
# Authors: 
# - Louis Lortie
# - Sepehr Moalemi
# - Syed Shabbir Ahmed
# %---------------------------------------------- Import Classes ---------------------------------------------------% #             
import numpy                as np
from Classes import Model   as Mdl
# %------------------------------------------------- Test Class ---------------------------------------------------% #
class Test:
    def __init__(self, tDataSet, methods, numFolds, 
                       alphas, lasos, 
                       max_iters, tols,
                       rmvSimilarRows=False,
                       writeTofile=False, filePath=''):
        # Store the Data Set
        self.dataSet = tDataSet
        
        # K-Fold Cross Validation
        self.numFolds = numFolds
        
        # Store hyper parameters
        self.methods   = methods
        self.alphas    = alphas
        self.lasos     = lasos
        self.tols      = tols
        self.max_iters = max_iters

        # Remove Similar Observations
        self.rmvSimilarRows = rmvSimilarRows
        
        # Printing 
        self.line = "-" * 10
        self.div  = "=" * 15
        
        # Redirect output to file
        if (writeTofile):
            import datetime, sys
            now = datetime.datetime.now()
            sys.stdout = open(filePath, "a")
            print (self.div*2 + now.strftime("%Y-%m-%d %H:%M:%S") + self.div*2)
            
    # Purpose: Test the Impact of removing similar rows of observations
    # # Test only one set of hyper parameters
    def rmvRowImpact(self):
        laso = 0
        print(f'{self.div} Effect of Removing Identical Observations on Model Accuracy {self.div}')
        for method in self.methods:
            print(f'{self.line*2} {method[0]} {self.line*2}')
            tModel1 = Mdl.Model(self.dataSet, method, self.numFolds,
                                eval(self.alphas[0]), laso,
                                self.tols[0], self.max_iters[0],
                                rmvSimilarRows=True)
            tModel0 = Mdl.Model(self.dataSet, method, self.numFolds, 
                                eval(self.alphas[0]), laso,
                                self.tols[0], self.max_iters[0],
                                rmvSimilarRows=False)
            
            # Error Accross the full dataset      
            Y_pred1 = Mdl.Model.fit_eval(tModel1, tModel1.X, tModel1.X, tModel1.Y)
            err1    = Mdl.Model.evalAccuracy(tModel1.Y, Y_pred1, verbos=False)
            
            Y_pred0 = Mdl.Model.fit_eval(tModel0, tModel0.X, tModel0.X, tModel0.Y)
            err0    = Mdl.Model.evalAccuracy(tModel0.Y, Y_pred0, verbos=False)
            
            Mdl.Model.printErrors(err0,"Not Removed", err1,"Removed")
            print("")
    
    # Purpose: Test the Impact of Removing Each Individual Feature
    def rmvFeature(self):
        laso = 0
        print(f'{self.div} Effect of Removing Each Individual Feature on Model Accuracy {self.div}')
        
        tModel = Mdl.Model(self.dataSet, [], self.numFolds,
                           eval(self.alphas[0]), laso,
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        indx = Mdl.Model.k_Fold_Gen_indx(tModel.X)
        
        for method in self.methods:
            print(f'\n{self.line*2} {method[0]} {self.line*2}')
            tModel.method = method
            
            # Error Without Removing any
            err0 = Mdl.Model.k_FoldCrossVal(tModel, tModel.X, tModel.Y, indx)
            
            # Remove each feature one by one and compare
            nf = tModel.cols - 1
            err = np.zeros((nf, 5))
            for f in range(0,nf):
                # Remove indx f
                X      = Mdl.Model.rmFeaturesByIndx(tModel.X, f+1)
                err[f] = Mdl.Model.k_FoldCrossVal(tModel, X, tModel.Y, indx)
            
            # Print Errors
            Test.printFeatureErrCol(err0, err, nf)
      
    # Purpose: Test the Impact of Removing Combination of Features
    def rmvCombFeatures(self, combins):
        laso = 0
        print(f'{self.div} Effect of Removing a Combination of Features on Model Accuracy {self.div}')
        
        tModel = Mdl.Model(self.dataSet, [], self.numFolds,
                           eval(self.alphas[0]), laso,
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        indx = Mdl.Model.k_Fold_Gen_indx(tModel.X)
        for method in self.methods:
            print(f'\n{self.line*2} {method[0]} {self.line*2}')
            tModel.method = method
            
            # Error Without Removing any
            err0 = Mdl.Model.k_FoldCrossVal(tModel, tModel.X, tModel.Y, indx)
            
            # Remove each Combination
            nf = len(combins)
            err = np.zeros((nf, 5))
            for i, combin in enumerate(combins):
                # Remove indx f
                X      = Mdl.Model.rmFeaturesByIndx(tModel.X, combin)
                err[i] = Mdl.Model.k_FoldCrossVal(tModel, X, tModel.Y, indx)
            
            # Print Errors
            Test.printFeatureErrCol(err0, err, nf, combins)
        
    # Purpose: Test the Impact of Adding Feature Compexity
    def addFeatureComplexity(self, flist):
        laso = 0
        print(f'{self.div} Effect of Adding Feature Complexity on Model Accuracy {self.div}')
        
        tModel = Mdl.Model(self.dataSet, [], self.numFolds,
                           eval(self.alphas[0]), laso,
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        indx = Mdl.Model.k_Fold_Gen_indx(tModel.X)
        for method in self.methods:
            tModel.method = method
            for fun in flist:
                print(f'\n{self.line*2} {method[0]} {self.line*2}')
                print(f'{self.line*2} {fun} {self.line*2}')
                
                # Error Without Any mofications
                err0   = Mdl.Model.k_FoldCrossVal(tModel, tModel.X, tModel.Y, indx)
                
                # Remove each Combination
                nf = tModel.cols - 1
                err = np.zeros((nf, 5))
                for f in range(0,nf):
                    # Append fun(f)
                    X      = Mdl.Model.addModFeature(tModel.X, f+1, eval(fun))
                    err[f] = Mdl.Model.k_FoldCrossVal(tModel, X, tModel.Y, indx)
                
                # Print Errors
                Test.printFeatureErrCol(err0, err, nf)
                
    # Purpose: Test the Impact of Adding a Set of Feature Compexities
    def addFeatureComplexitySet(self, flist):
        laso = 0
        print(f'{self.div} Effect of Adding a Set of Feature Complexities on Model Accuracy {self.div}')
        
        tModel = Mdl.Model(self.dataSet, [], self.numFolds,
                           eval(self.alphas[0]), laso,
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        indx = Mdl.Model.k_Fold_Gen_indx(tModel.X)
        for method in self.methods:
            tModel.method = method
            print(f'\n{self.line*2} {method[0]} {self.line*2}')
            
            # Error Without Any mofications
            err0 = Mdl.Model.k_FoldCrossVal(tModel, tModel.X, tModel.Y, indx)
            
            # Add Each Combination
            nf = len(flist)
            err = np.zeros((nf, 5))
            for i, combin in enumerate(flist):
                # Append fun(f)
                X = Mdl.Model.addModFeature(tModel.X, combin[0][0], eval(combin[0][1]))
                X = Mdl.Model.addModFeature(X, combin[1][0], eval(combin[1][1]))
                if(len(combin) == 3):
                    X = Mdl.Model.addModFeature(X, combin[2][0], eval(combin[2][1])) 
                err[i] = Mdl.Model.k_FoldCrossVal(tModel, X, tModel.Y, indx)
            
            # Print Errors
            Test.printFeatureErrCol(err0, err, nf)               
                
    # Purpose: Test the Impact of Stepsize and Laso const for GD
    def findOptimalAlphaLambda(self):       
        print(f'{self.div} Effect of Stepsize and Laso Const on Model Accuracy {self.div}')
        tModel = Mdl.Model(self.dataSet, [], self.numFolds,
                           eval(self.alphas[0]), 0,
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        indx = Mdl.Model.k_Fold_Gen_indx(tModel.X)
        for method in self.methods:
            tModel.method = method
            print(f'\n{self.line*2} {method[0]} {self.line*2}')
            for alpha in self.alphas:
                print(f'\n{self.line*2} {alpha} {self.line*2}')
                tModel.alpha = eval(alpha)
                nf  = len(self.lasos)
                err = np.zeros((nf, 5))
                for i,laso in enumerate(self.lasos):
                    tModel.laso = laso
                    err[i] = Mdl.Model.k_FoldCrossVal(tModel, tModel.X, tModel.Y, indx)
                rows = np.floor_divide(nf, 3)
                rem  = nf - rows*3
                for f in range(0,rows):
                    f=3*f
                    Mdl.Model.printErrors(err[f],   "Lambda = "+str(self.lasos[f]), 
                                            err[f+1], "Lambda = "+str(self.lasos[f+1]),
                                            err[f+2], "Lambda = "+str(self.lasos[f+2]))
                print("")
                f = rows*3
                if(rem == 2):
                    Mdl.Model.printErrors(err[f],   "Lambda = "+str(self.lasos[f]), 
                                            err[f+1], "Lambda = "+str(self.lasos[f+1]))
                elif(rem == 1):
                    Mdl.Model.printErrors(err[f],   "Lambda = "+str(self.lasos[f]))  
                print("")  
        
    # Purpose: Test the Impact OF tol and max_iter number for GD
    def findOptimalTolMaxIter(self):
        print(f'{self.div} Effect of Tol and Max Iter on Model Accuracy {self.div}')
        tModel = Mdl.Model(self.dataSet, [], self.numFolds,
                           eval(self.alphas[0]), self.lasos[0],
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        indx = Mdl.Model.k_Fold_Gen_indx(tModel.X)
        for method in self.methods:
            tModel.method = method
            print(f'\n{self.line*2} {method[0]} {self.line*2}')
            for tol in self.tols:
                print(f'\n{self.line*2}Tol = {tol} {self.line*2}')
                tModel.tol = tol
                nf  = len(self.max_iters)
                err = np.zeros((nf, 5))
                for i,imax in enumerate(self.max_iters):
                    tModel.max_iter = imax
                    err[i] = Mdl.Model.k_FoldCrossVal(tModel, tModel.X, tModel.Y, indx)
                rows = np.floor_divide(nf, 3)
                rem  = nf - rows*3
                for f in range(0,rows):
                    f=3*f
                    Mdl.Model.printErrors(err[f],   "Max_Iter = "+str(self.max_iters[f]), 
                                          err[f+1], "Max_Iter = "+str(self.max_iters[f+1]),
                                          err[f+2], "Max_Iter = "+str(self.max_iters[f+2]))
                print("")
                f = rows*3
                if(rem == 2):
                    Mdl.Model.printErrors(err[f],   "Max_Iter = "+str(self.max_iters[f]), 
                                          err[f+1], "Max_Iter = "+str(self.max_iters[f+1]))
                elif(rem == 1):
                    Mdl.Model.printErrors(err[f],   "Max_Iter = "+str(self.max_iters[f]))  
                print("")   
                  
    # Purpose: Run the Model on the full Dataset
    def findErrOnAll(self, dataSet):
        print(f'{self.div} Modified Model Error Over the Entire Training Data {self.div}')
        tModel = Mdl.Model(self.dataSet, self.methods[0], self.numFolds,
                           eval(self.alphas[0]), self.lasos[0],
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        tModel1 = Mdl.Model(dataSet, self.methods[0], self.numFolds,
                           eval(self.alphas[0]), self.lasos[0],
                           self.tols[0], self.max_iters[0], 
                           self.rmvSimilarRows)
        Y_pred = Mdl.Model.fit_eval(tModel, tModel.X, tModel.X, tModel.Y)
        Y_pred1 = Mdl.Model.fit_eval(tModel1, tModel1.X, tModel1.X, tModel1.Y)
        err    = Mdl.Model.evalAccuracy(tModel.Y, Y_pred, verbos=False)
        err1    = Mdl.Model.evalAccuracy(tModel1.Y, Y_pred1, verbos=False)
        Mdl.Model.printErrors(err, "Modified", err1, "No Modifications")
        
    # Purpose: Print errors in columns
    @staticmethod
    def printFeatureErrCol(err0, err, nf, *arg):
        txt = lambda f : f
        if(len(arg) != 0):
            arg = arg[0]
            txt = lambda f: arg[f-1]
            
        rows = np.floor_divide(nf, 3)
        rem  = nf - rows*3
        for f in range(0,rows):
            f=3*f
            Mdl.Model.printErrors(err[f],"Feature " + str(txt(f+1)), 
                                  err[f+1],"Feature " + str(txt(f+2)),
                                  err[f+2],"Feature " + str(txt(f+3)))
            print("")
        f = rows*3
        if(rem == 2):
            Mdl.Model.printErrors(err[f],  "Feature " + str(txt(f+1)), 
                                  err[f+1],"Feature " + str(txt(f+2)),
                                  err0, "No Modifications")
        elif(rem == 1):
            Mdl.Model.printErrors(err[f],"Feature " + str(txt(f+1)),
                                  err0, "No Modifications")
        elif(rem == 0):
            Mdl.Model.printErrors(err0, "No Modifications")   
        print("")