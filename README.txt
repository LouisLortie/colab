--------------------------------------------
# ECSE 551 - A1
# Authors: 
# - Louis Lortie
# - Sepehr Moalemi
# - Syed Shabbir Ahmed
--------------------------------------------
============ Folder Breakdown ============
./A1_Main.py
- Purpose:
- - Generate a Model for each File
- - Through a series of tests, modify the model/data set
- - 1) Checks if there are repeated rows in the data set
- - - Compares results with/without removing the rows
- - - - Stores Optimal Case
- - 2) Applying different stepsize functions to the gradient decent
- - - Compares various Lasso Regurulization constants
- - - - Stores Optimal alpha and lambda
- - 3) Runs gradient decent with various tol and max_iter values
- - - - Stores Optimal tol and max_iter
- - 4) Removes each feature one at a time to see effect on accuracy
- - - - Stores features that improve accuracy when removed
- - 5) Removes each combination of the features above
- - - - Stores optimal combination
- - 6) Applies a series of function to each feature and appends it to the dataset
- - - - Stores feature + functions that improve accuracy
- - 7) Add combination of these feature + functions
- - - - Store the set that imrpoves accuracy
- - 8) Finally, apply all stored findings to the dataset
- - - - Compare the accuracy to a model with no modifications

- Usage:
- - The Main File to Run
- - All parameters can be modified 
- - To see the result of each unit test, set line 14: {verbos = True}
- - To output result to a file instead of terminal, set line 15: {writeTofile = True}
- - To see dataset plots, set line 20: {visualizeData = True}

- Note: 
- - Results of tests are commented under them
