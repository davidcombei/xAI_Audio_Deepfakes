import sys, os
import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from config_ma import *

print(f"Using layer {L} from BERT")


dbs = {'asv19':0, 'asv21':1, 'timit':2,  'mlaad':3,  'asv5':4}
BONAFIDES = False
print(f"Using BONAFIDES={BONAFIDES}\n")

FEATURE_MODEL = "facebook/w2v-bert-2.0"

CKPT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "disertatie_demo", "checkpoints", "knn_model_attribution.joblib",
)


Y = []
valid = []
c = 0
for fi in metadata[:]:
    with open(os.path.join(meta_dir,fi)) as fin:
        for line in sorted(fin.readlines()):
            if fi.split('_')[0] in dbs:
                if BONAFIDES:
                    #Y.append(line.strip().split('|')[2])
                    Y.append(fi.split('_')[0]+"_"+line.strip().split('|')[2])

                else:
                    if 'bonafide' not in line.strip().split('|')[2]:
                        #Y.append(line.strip().split('|')[2])
                        Y.append(fi.split('_')[0]+"_"+line.strip().split('|')[2])
                        valid.append(c)
            c+=1

## Assign numeric labels to systems
systems = {k:i for i, k in enumerate(sorted(set(Y)))}
print("Number of systems:", len(systems))
target_names = list(systems.keys())
print(target_names)

Y = np.array([systems[Y[i]] for i in range(len(Y))])
print("Y shape: ", Y.shape)

X = []
for fi in feats [:]:
    x = np.load(os.path.join(feats_dir, fi))
    print(fi, x.shape)
    X.extend(x)

X = np.array(X)
if not BONAFIDES:
    X = X[valid]

print ("\nX shape:", X.shape)


## Train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train size: ", Xtrain.shape[0])
print("Test size: ", Xtest.shape[0])


print("\nScaling...")
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)


K = 11
print("Fitting...")
clf = KNeighborsClassifier(n_neighbors=K)#, weights='distance')
print("\n")
print(clf)
clf.fit(Xtrain, ytrain)


print("Predicting Xtest...")
Y_hat = clf.predict(Xtest)
print(classification_report(ytest, Y_hat, target_names=target_names))


print("\nRefitting scaler + kNN on the full dataset...")
final_scaler = StandardScaler()
Xall = final_scaler.fit_transform(X)
final_clf = KNeighborsClassifier(n_neighbors=K)
final_clf.fit(Xall, Y)

bundle = {
    "knn": final_clf,
    "scaler": final_scaler,
    "labels": {i: k for k, i in systems.items()},
    "systems": systems,
    "target_names": target_names,
    "layer": L,
    "K": K,
    "feature_model": FEATURE_MODEL,
    "n_systems": len(systems),
    "n_train": int(X.shape[0]),
}

os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
joblib.dump(bundle, CKPT_OUT, compress=3)
print(f"Saved kNN attribution checkpoint -> {CKPT_OUT}")
print(f"  systems: {len(systems)} | exemplars: {X.shape[0]} | layer: {L} | K: {K}")

exit(0)
with open("results_model_attribution_KNN_21_ALL.txt", 'w') as fout:
    for k in systems:
        fout.write(f"{k}:{systems[k]}\n")
    fout.write("------\n")

    fout.write(f"Training data: {Xtrain.shape[0]} | Test data: {Xtest.shape[0]} \n")
    fout.write("------\n")

    fout.write(classification_report(ytest, Y_hat))
    fout.write("------\n")

    np.savetxt(fout, confusion_matrix(ytest, Y_hat, normalize='true'), delimiter=',', fmt='%.2f')
