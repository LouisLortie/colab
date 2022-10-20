    # Generate Model for each file
    for file in dataFiles:
        # Generate Model from list of CSV Files
        tDataSet = Dset.Dataset(file)
        if (visualizeData): 
            tDataSet.analyzeData(pltStd, pltMean, pltSpread, pltHist)
            
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
                if(method[1] == "GD"):
                    tModel.alpha = alpha
                    print(f'\n{i+1}.{j+1}) Run {method[0]} Model for {tDataSet.name}')
                    alphastr = str(inspect.getsourcelines(alphas[j])[0][0])
                    print(f'Using{alphastr}')
                else:
                    print(f'\n{i+1}) Run {method[0]} Model for {tDataSet.name}')
                    
                # K-Fold Cross Validation
                err = Mdl.Model.k_FoldCrossVal(tModel, X, Y, verbos=False)
                print(f'Accros {numFolds}-Fold Cross Validation\n')
                Mdl.Model.printErrors(err)
                
                # Error Accross the full dataset
                Y_pred = Mdl.Model.fit_eval(tModel, X, X, Y)
                Mdl.Model.evalAccuracy(Y, Y_pred, verbos=False)
    