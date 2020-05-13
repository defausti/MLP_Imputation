# MLP_Imputation
Univariate Imputation with neural network


the Software is written in python using the follows main modules:
pandas=0.24.1
keras=2.2.4

you can install python anaconda3 and use conda package manager
conda install keras=2.2.4 pandas=2.2.4




Run the from Code directory

this program wants a kfold validation framework
in the example datafile (DATASET_WITH_FOLDS.txt) there is a field FOLD with index 0,1,2,3,4


You have to run Imputation.py with different arguments in this order:

1. To prepare training e test file :

python Imputation.py 0 PREPROC

the fold index 0 means that:
 records with index fold [1,2,3,4] build the training set 
 records with index fold 0 bultd the test set




2. To train the MLP model on training set 

python Imputation.py 0 TRAINING


3. To impute the TARGET VARIABLE on testset 
python Imputation.py 0 IMPUTATION

in Data/Imputation directory you find the output file 
with the target variable imputated with modal approach and distributional approach
The imputation target variable is calculated on the test set created during the PREPROC phase


