from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import torch

town01 = torch.load("data/shift/processed/town01_semseg.pt")
town02 = torch.load("data/shift/processed/town02_semseg.pt")

label_names = [
    "unlabeled",
    "building",
    "fence",
    "other",
    "pedestrian",
    "pole",
    "roadLine",
    "road",
    "sideWalk",
    "vegetation",
    "vehicles",
    "wall",
    "trafficSign",
    "sky",
    "ground",
    "bridge",
    "railtrack",
    "guardrail",
    "trafficlight",
    "static",
    "dynamic",
    "water",
    "terrain",
]

town01_labels = torch.ones(town01.shape[0])
town02_labels = torch.zeros(town02.shape[0])

X = torch.cat([town01, town02])
y = torch.cat([town01_labels, town02_labels])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# lsvc = LinearSVC(C=0.005, penalty="l1", dual=False).fit(X_train, y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_train = model.transform(X_train)
# X_test = model.transform(X_test)
print("train shape:", X_train.shape)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

print(tree.export_text(clf, feature_names=label_names))
