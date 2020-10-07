# Long-Term Risk Stratification of Liver Transplant Recipients: Real-time Application of Deep Learning Algorithms on Longitudinal Data



## Importance

The long-term survival of **liver transplant** recipients beyond one year is significantly compromised by an increased risk of **cancer**, **cardiovascular mortality**, **infection** and **graft failure**. There are currently limited clinical tools to identify patients at risk of these complications, which would flag them for screening tests and life-saving interventions. 

We hereby propose **Deep Learning** models designed for longitudinal data that reliably predicts an updated clinical outlook for individual patients. Here is an example of how our top-performing **Transformer** model estimates the risk progression of a patient more than 20 years post-transplant, accurately outlining the top complications.

<img src="https://github.com/bowang-lab/Transplant_Time_Series/blob/master/Img/patient.png" width="60%" height="60%">

## Requirements and Installation
First, install dependencies
```bash
# clone project   
git clone git@github.com:bowang-lab/Transplant_Time_Series.git

# install project   
cd Transplant_Time_Series
conda env create -f environment.yml
```
Next, activate the environment and run it. 
 ```bash
conda activate transplant

# run module: 
python 

# check `Models/Transformer/train.py` for more args
```

## Methods

A DL-based **Transformer** model was developed and trained on a set of 42,146 LT recipients (median age 53, IQR 45-59 years; 40.8% women) from the publicly available Scientific Registry of Transplant Recipients (SRTR). 

The transferability of the model was further evaluated by testingfine-tuning on a local dataset from University Health Network in Toronto, Canada, consisting of 3,269 patients (median age 54, IQR 46-61 years; 33.0% women).

<img src="https://github.com/bowang-lab/Transplant_Time_Series/blob/master/Img/transformer.png" width="80%" height="80%">



## Results

The area under the receiver operating characteristic curve (AUROC) for the top-performing **Transformer** Model across all outcomes in the SRTR dataset was 0.804, 99% CI [0.795, 0.854] (1 year) and 0.733, 99% CI [0.729, 0.769] (5 years). In the UHN dataset, the top deep learning AUROC was 0.807, 99% CI [0.795, 0.842] (1 year) and 0.722, 99% CI [0.705, 0.764] (5 years). AUROCs ranged from 0.695 for 5-year infection death to 0.856 for 1-year graft failure.

<img src="https://github.com/bowang-lab/Transplant_Time_Series/blob/master/Img/results.png" width="60%" height="60%">

We compared the performance of our **Transformer** model with other DL-based models (Temporal Convolutional Network model, Recurrent Neural Network model) as well as traditional logistic regression (LR) models. The Transformer model outforms the LR models by significant margin.

<img src="https://github.com/bowang-lab/Transplant_Time_Series/blob/master/Img/comparison.png" width="70%" height="70%">

