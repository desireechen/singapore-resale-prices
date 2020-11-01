from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

class Model:

    def __init__(self):
        # initialise model.
        self.model = KNeighborsClassifier(n_neighbors=5)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Fit Random Forest Classifier.        
        :param params: The parameters used to train the classifier.
        :param X_train: The training input examples.
        :param y_train: The target values. An array of int.
        :return: Train F1 score as a single float.
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return print('The Test accuracy score for K Neighbors Classifier is {}' .format(score))