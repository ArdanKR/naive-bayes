from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score,ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split

# module import endline #

data=pd.read_csv('./dataset.csv',chunksize=100000,index_col=0) # import dataset(csv)
df = pd.concat(data)

print ("Balanced Dataset shape: ")
print(df.shape)
total = len(df)*1.

ax=sns.countplot(x=" Label", data=df)
for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

ax.yaxis.set_ticks(np.linspace(0, total, 2))
ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
plt.title('Label Distribution (Balanced Dataset)')
plt.show()

num_cols = df.select_dtypes(exclude=['object']).columns
fwd_cols = [col for col in num_cols if 'Fwd' in col]
bwd_cols = [col for col in num_cols if 'Bwd' in col]
def getCorrelatedFeatures(corr):
    correlatedFeatures = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if (abs(corr.iloc[i, j])) > 0.95:
                print(corr.columns[i],corr.iloc[i,j])
                correlatedFeatures.add(corr.columns[i])
    return correlatedFeatures

corr = df[fwd_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool_))
plt.subplots(figsize=(20,20))
sns.heatmap(corr, annot=True, mask=mask)
plt.show()

x=df[[' Flow Duration',' Total Fwd Packets', ' Total Backward Packets','Total Length of Fwd Packets', ' Total Length of Bwd Packets',' Fwd Packet Length Max', ' Fwd Packet Length Min',' Fwd Packet Length Mean', ' Fwd Packet Length Std','Bwd Packet Length Max', ' Bwd Packet Length Min',' Bwd Packet Length Mean', ' Bwd Packet Length Std','FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', 'Idle Mean', ' Idle Std',' Idle Max', ' Idle Min']]
y=df[' Label']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.18, shuffle=False, random_state=1004)

gnb = GaussianNB() # Gaussian NBC
gnb_fit = gnb.fit(X_train, y_train) # Train model
y_pred = gnb_fit.predict(X_test) # Test and Score
print(y_pred)
cm=confusion_matrix(y_test, y_pred) # Confusion Matrix
print(cm)
print(classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test,  y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Benign","DDoS"])
disp.plot(cmap='Blues')
plt.title('RF Confusion Matrix')
plt.show()