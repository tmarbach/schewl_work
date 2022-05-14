# Set-up 
## Conda YAML
For those with Anaconda configured in their enviroment please use the Snek-Conda-Install.yml

# Manual Installation
Otherwise Installation includes the following:
 - Base Anaconda Installation with Python v3.8
 - Imbalanced Learn v0.9.0 (See URL for installation via Anaconda console)
   - https://anaconda.org/conda-forge/imbalanced-learn?msclkid=bb462899cdb211ec8bab48c64b8838cf
 - openpyxl v3.0.9 opening excel files
 - pyarrow v4.0.1 interoperability with pandas and numpy
 - xlsxwriter v3.03 writing to excel files
 - ipykernel v6.13.0 viewing the data visualization notebook
 - matplotlib v3.5.1

# Viewing Data Exploration
The data visualizations of the intial exploration into the Snake Dataset are found in the data_exploration notebook.

# Running Model
Run modeling.py via the command line or the launch.json via VScode.

## Optional Arguments
 -m : Specify ML model
    options
      rf: Random Forest
      nb: Naive Bayes
      svm: Support Vector Machines
    default
      Random Forest
 -o : The sampling type
    options
      o: oversampling
      s: SMOTE
      a: ADASYN
    default
      ns: no sampling applied

## Running the code
### VS Code
Change the flags in the args list in the launch.json.
The following is an example of running and SVM model with a SMOTE sampling technique.
  "args": [
    "-m", "svm",
     "-o", "s"]

### Run Via Command Line
#### With default paramters
./model.py 

#### With paramters
Example of running with a Naive Bayes modeling with an Oversampling technique
./model.py -m nb -o o



