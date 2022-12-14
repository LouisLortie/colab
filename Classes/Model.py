# ECSE 551 - A1
# Authors: 
# - Louis Lortie
# - Sepehr Moalemi
# - Syed Shabbir Ahmed
# %-------------------------------------------------- Packages -----------------------------------------------------% #
import numpy                as np
from Classes import Dataset as Dset
# %------------------------------------------------- Model Class ---------------------------------------------------% #
class Model:
    def __init__(self, dataSet, method, K, 
                       alpha, laso=0.0, 
                       tol=1e-5, max_iter=1e+3, rmvSimilarRows=False):
        # Store the Data Set
        self.dataSet = dataSet
        
        # Remove Similar Observations if any
        if(rmvSimilarRows):
            self.X, self.Y = Model.rmvSimilarObs(dataSet.data[:,:-1], 
                                                 dataSet.data[:,-1].reshape(-1,1))
        else:
            self.X, self.Y = dataSet.data[:,:-1], dataSet.data[:,-1].reshape(-1,1)
        
        # Add Bias Term to Data Set
        self.X = np.c_[np.ones(np.shape(self.X)[0]), self.X]
        
        # Model training data dim
        (self.rows, self.cols) = np.shape(self.X)
        
        # Linear Classifier Method
        self.method = method
        
        # Step size for Gradient Decent
        self.alpha = alpha
        
        # Lambda for laso regression
        self.laso = laso
        
        # K-Fold Cross Validation
        self.K = K
        
        # Convergence Tol and Max Number of iterations
        (self.tol, self.max_iter) = tol, max_iter

    # %------------------------------------------- Model Methods ---------------------------------------------------% #
    # Purpose: Train model using training data X and labels Y
    # # Accepts methods
    @staticmethod
    def fit(model, X, Y):
        match model.method:
            case [_, "LS"]:
                return Model.LS(X, Y)
            
            case [_, "GD", type]:
                if (type == "LinReg" or type == "LogReg"):
                    return Model.GD_Solver(model, X, Y, model.alpha, type)
            
            case [_, "DA", type]:
                if(type == "Linear" or type == "Quadratic"):
                    return Model.DA_Solver(model, X, Y, type)
                
            case other: 
                raise NotImplementedError(f'Method Not Implemented')       
    
    # Purpose: Evaluate model given test data
    @staticmethod
    def eval(model, X_new, args):
        match model.method:
            case [_, "LS"]:
                return Model.threshold(X_new @ args, 0.5)
            
            case [_, "GD", type]:
                if (type == "LinReg"):
                    return Model.threshold(X_new @ args, 0.5)
               
                if (type == "LogReg"):
                    return Model.threshold(X_new @ args, 0)
                
            case [_, "DA", type]:
                if(type == "Linear" or type == "Quadratic"):
                    delta0, delta1 = args
                    return Model.threshold(delta1(X_new).flatten(), 
                                           delta0(X_new).flatten())
                  
            case other: 
                raise NotImplementedError(f'Method Not Implemented')
    
    # Purpose: Fit and Evalute
    @staticmethod
    def fit_eval(model, X, X_new, Y):
        return Model.eval(model, X_new, Model.fit(model, X, Y))
        
    # %--------------------------------- Linear Classifier Logistic Regression -------------------------------------% #
    # Purpose: Output Threshloding given a Cut-Off
    @staticmethod
    def threshold(Y, cutOff):
        def f(y1, y0):
            if y1 > y0:
                return 1
            if y1 < y0:
                return 0
            else:
                return np.random.choice([0,1]) 
        if (np.isscalar(cutOff)):
            return np.array([f(y1, cutOff) for y1 in Y])
        return np.array([f(y1, y0) for y1, y0 in zip(Y, cutOff)])
    
    # Purpose: Least Squares Linear-Regression O(m^3 + nm^2)
    # # Calculates Pseudo-Inverse of X.T*X
    # # {y_new = x_new*w} And {w = inv(X.T*X)*X.T*Y}
    @staticmethod
    def LS(X, Y):
        return np.linalg.pinv(X.T @ X) @ (X.T @ Y)

    # Purpose: Gradient Descent General Solver
    # # By defualt uses LS to find w_0
    # # alpha is a func that produces a Robbins-Monroe Sequence 
    @staticmethod
    def GD_Solver(model, X, Y, alpha, type, W=[]):
        # Purpose: Gradient Descent Linear-Regression
        # {W = 2 * X.T @ (X @ W - Y)}
        GD_LinReg = lambda X, Y, W : 2 * (X.T @ (X @ W - Y))
        
        # Purpose: Gradient Descent Logistic-Regression
        # SUM {-X * (Y - Model.sigma(X @ W))}
        sigma = lambda a : 1/(1 + np.exp(-a))
        def GD_LogReg(X, Y, W):
            delta = -X * (Y - sigma(X @ W))   
            return delta.sum(axis=0).reshape(-1,1)
        
        # Set w_0 using LS if no w_0 given
        if (not W):
            W = np.zeros((np.shape(X)[1], 1))
            # W = Model.LS(X, Y)
        
        # Set delta based on type
        delta = GD_LinReg
        if(type == "LogReg"):
            delta = GD_LogReg    
    
        # Initialize counter
        k  = 0
        dW = np.zeros_like(W) + 2*model.tol
        
        while(model.tol < np.linalg.norm(dW, ord=2) and k < model.max_iter):
            dW = alpha(k) * (delta(X, Y, W) + model.laso * np.sign(W))
            W -= dW
            k += 1
            
        # if(k == model.max_iter):
        #     print(f'Never reached tol = {model.tol}')
        # else:
        #     print(f'Reached tol = {model.tol} after {k} Iterations')
        
        return W
    
    # Purpose: Discriminant Analysis (DA)
    @staticmethod
    def DA_Solver(model, X, Y, type):
        dataSet = Dset.Dataset(np.c_[X, Y], model.dataSet.name)
        P0 = dataSet.Class0.rows/dataSet.rows
        P1 = dataSet.Class1.rows/dataSet.rows

        M0 = dataSet.Class0.colAvrg.reshape(-1, 1)
        M1 = dataSet.Class1.colAvrg.reshape(-1, 1)
        
        dX0 = (dataSet.Class0.X.T - M0)
        dX1 = (dataSet.Class0.X.T - M1)
        
        # Linear Discriminant Analysis (LDA)
        if (type == "Linear"):
            InvCov = np.linalg.pinv((dX0 @ dX0.T + dX1 @ dX1.T)/(dataSet.rows - 2))

            delta0 = lambda x : (x @ InvCov @ M0) - (M0.T @ InvCov @ M0)/2 + np.log(P0)
            delta1 = lambda x : (x @ InvCov @ M1) - (M1.T @ InvCov @ M1)/2 + np.log(P1)
            
            return (delta0, delta1)
          
        # Quadratic Discriminant Analysis (QDA) 
        if (type == "Quadratic"):
            InvCov0 = np.linalg.pinv((dX0 @ dX0.T)/(dataSet.Class0.rows - 1))
            InvCov1 = np.linalg.pinv((dX1 @ dX1.T)/(dataSet.Class1.rows - 1))

            delta0 = lambda x : (np.log(P0) - np.linalg.det(InvCov0))/2 - (x.T - M0).T @ InvCov0 @ (x.T - M0)/2
            delta1 = lambda x : (np.log(P1) - np.linalg.det(InvCov1))/2 - (x.T - M1).T @ InvCov1 @ (x.T - M1)/2
            
            return (delta0, delta1)
 
    # %--------------------------------------- Feature Set Modification --------------------------------------------% #
    # Purpose: Removes columns given by indx from the modified feature set
    @staticmethod
    def rmFeaturesByIndx(X, indx):
        return np.delete(X, indx, axis=1)  
        
    # Purpose: Appends a map of a column given by indx to the modified feature set
    @staticmethod
    def addModFeature(X, indx, f):
        return np.c_[X, np.array([f(xi) for xi in X[:,indx]])]

    # %------------------------------------ Observation Set Modification -------------------------------------------% #
    # # Purpose: Check if data set has two identical observations
    # -  Remove them if they belong to different classes
    @staticmethod
    def rmvSimilarObs(X, Y):
        # Find repeated rows
        unq_idx, unq_cnt = np.unique(X, return_inverse=True, 
                                        return_counts =True, axis=0)[1:3]
        cnt_mask = unq_cnt > 1
        
        # Return if no repeated rows found
        if(np.all(cnt_mask == False)):
            print(f'Model Data Set has no identical observation.')
            return X, Y
        
        # Find indx of repeated rows
        cnt_idx, = np.nonzero(cnt_mask)
        idx_mask = np.in1d(unq_idx, cnt_idx)
        idx_idx, = np.nonzero(idx_mask)
        srt_idx  = np.argsort(unq_idx[idx_mask])
        dup_idx  = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
        
        # Delete repeated rows
        dlt_idx  = np.array([], int)
        
        # Counters
        sameY = 0
        diffY = 0
        for set in dup_idx:
            if (np.all(Y[set] == Y[set][0])):
                # Keep one row if all are of the same class
                sameY+=1
                dlt_idx = np.append(dlt_idx, set[1:])
            else:
                # Remove all rows if they arent of the same class
                diffY+=1
                dlt_idx = np.append(dlt_idx, set)
                
        print(f'Removed a {sameY} Identical Observations in the same class from Data')
        print(f'Removed a {diffY} Identical Observations in diffrent classes from Data')
        
        return np.delete(X, dlt_idx, axis=0), np.delete(Y, dlt_idx, axis=0)
    
    # %----------------------------------------- Testing Accuracy  -------------------------------------------------% #
    # Purpose: K-Fold Cross Validation
    # # [OPTION] Verbos = True to print the results
    @staticmethod
    def k_Fold_Gen_indx(X):
        # Generate Random Indices
        rows = np.shape(X)[0]
        indx = np.arange(rows)
        np.random.shuffle(indx)
        return indx
    @staticmethod
    def k_FoldCrossVal(model, X, Y, indx, verbos=False):
        rows = np.shape(X)[0]
        # Determine number of observations needed for k-folds
        bucketSize = np.floor_divide(rows, model.K)
        
        # Trim the remaining indices
        indx = indx[0:bucketSize * model.K]
        
        # Iterate through the experiments
        err = np.zeros(5)
        for k in range(model.K):
            bucket    = np.arange(k * bucketSize, (k+1) * bucketSize)
            valIndx   = indx[bucket]
            trainIndx = np.delete(indx, bucket)
            
            # Fit and evaluate bucket
            Y_pred = Model.fit_eval(model, X[trainIndx], X[valIndx], Y[trainIndx])
            
            # Find error
            if(verbos): 
                print(f'\nExperiment {k}:')
            err += Model.evalAccuracy(Y[valIndx], Y_pred, verbos)
        
        # Return avrg error
        return err/model.K
    
    # Purpose: Evaluate Accuracy of Model given predictions
    # # [OPTION] Verbos = True to print the results
    @staticmethod
    def evalAccuracy(Y, Y_predicted, verbos=False):
        FP, FN = 0, 0
        TP, TN = 0, 0
        for t, p in zip(Y, Y_predicted):
            if (t == 1):
                if(p == 0):
                    FN += 1
                else:      
                    TP += 1
            elif (t == 0):
                if(p == 1): 
                    FP += 1
                else:       
                    TN += 1
                
        # Convert to percentage 
        rows = np.shape(Y)[0]
        FP  /= rows
        FN  /= rows 
        
        accuracy = 1 - (FP + FN)
        error    = 1 - accuracy
        
        recall    = (TP + 1) / (TP + FN + 2)
        precision = (TP + 1) / (TP + FP + 2)
        
        F1_Measure = 2 * recall * precision / (recall + precision)
        
        Err = [FP, FN, F1_Measure, accuracy, error]
        
        if (verbos):
            Model.printErrors(Err, "Percentage")
        return Err

    # Purpose: Print Errors
    @staticmethod
    def printErrors(*arg):
        errs = ['False Positive Rate:',
                'False Negative Rate:',
                'F1 Measure:',
                'Accuracy:',
                'Error:']

        # printing Aligned Header
        if(len(arg) == 2):
            Err1, txt1 = arg
            print(f"{'Errors' : <20}{txt1 : ^20}")
            for i in range(0, 5):
                print(f"{errs[i] : <20}{Err1[i]*100:^20.3f}")
        if(len(arg) == 4):
            Err1, txt1, Err2, txt2 = arg
            print(f"{'Errors' : <20}{txt1 : ^20}{txt2 : ^20}")
            for i in range(0, 5):
                print(f"{errs[i] : <20}{Err1[i]*100:^20.3f}{Err2[i]*100:^20.3f}")
        if(len(arg) == 6):
            Err1, txt1, Err2, txt2, Err3, txt3 = arg
            print(f"{'Errors' : <20}{txt1 : ^20}{txt2 : ^20}{txt3 : ^20}")
            for i in range(0, 5):
                print(f"{errs[i] : <20}{Err1[i]*100:^20.3f}{Err2[i]*100:^20.3f}{Err3[i]*100:^20.3f}")