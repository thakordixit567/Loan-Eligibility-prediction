import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('loan_data.csv')
#print(df.head())

#print(df.shape)

#print(df.info())

#print(df.describe())

#temp = df['Loan_Status'].value_counts()
#plt.pie(temp.values,labels=temp.index,autopct='%1.1f%%')
#plt.show()

plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
	plt.subplot(1, 2, i+1)
	sb.countplot(data=df, x=col, hue='Loan_Status')
plt.tight_layout()
plt.show()
