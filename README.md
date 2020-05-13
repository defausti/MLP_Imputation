# MLP_Imputation
Univariate Imputation with neural network

<br><br>
the Software is written in python using the follows main modules:<br>
pandas=0.24.1<br>
keras=2.2.4<br>
you can install python anaconda3 and use conda package manager<br>
conda install keras=2.2.4 pandas=2.2.4<br>
<br><br>
<br><br>
Run the from Code directory<br>
this program wants a kfold validation framework<br>
in the example datafile (DATASET_WITH_FOLDS.txt) there is a field FOLD with index 0,1,2,3,4<br>
<br><br>
<br><br>
You have to run Imputation.py with different arguments in this order:<br>
1. To prepare training e test file :<br>
python Imputation.py 0 PREPROC<br>
<br>
the fold index 0 means that:<br>
-records with index fold [1,2,3,4] build the training set<br>
-records with index fold 0 bultd the test set<br>
<br><br>
<br><br>
2. To train the MLP model on training set <br>
python Imputation.py 0 TRAINING<br>
<br><br>
<br><br>
3. To impute the TARGET VARIABLE on testset <br>
python Imputation.py 0 IMPUTATION<br>
<br>
in Data/Imputation directory you find the output file <br>
with the target variable imputated with modal approach and distributional approach<br>
The imputation target variable is calculated on the test set created during the PREPROC phase<br>


