from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import csv
import pickle

classifiers = {
    "Nearest_Neighbors": KNeighborsClassifier(3),
    "Random_Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural_Net": MLPClassifier(alpha=1),
    "AdaBoost": AdaBoostClassifier(),
    "Naive_Bayes": GaussianNB(),
}

def perform_analysis(X, y, test_prop, START_TIME):
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=test_prop)
    results = []
    for name, classifier in classifiers.iteritems():
        classifier.fit(X_training, y_training)
        score = classifier.score(X_testing, y_testing)
        results.append((name, classifier, score))
        with open(START_TIME + "/" + START_TIME + '_' + name + '.pickle', 'wb') as file:
            pickle.dump(classifier, file)
    with open(START_TIME + "/" + START_TIME + '_classifier_analysis.csv', 'wb') as file:
        file_writer = csv.writer(file)
        for name, classifier, score in results:
            file_writer.writerow((name, score))   
    return results
