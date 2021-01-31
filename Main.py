import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation


# from sklearn.tree import export_graphviz
# from IPython.display import Image
# import pydotplus
#

def faz_one():
    pol_fake = pd.read_csv("pol-fake.csv")
    pol_real = pd.read_csv("pol-real.csv")
    pol_real["real"] = 1
    pol_fake["real"] = 0

    result = pd.concat([pol_real, pol_fake])

    shuffled_resutl = result.iloc[np.random.permutation(len(result))].reset_index(drop=True)

    lable_name = list(shuffled_resutl.columns)

    feature_cols = lable_name[:-1]

    # split dataset in features and target variable
    x = shuffled_resutl[feature_cols]  # Features
    y = shuffled_resutl.real  # Target variable

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=1)  # 80% training and 20% test

    return [X_train, X_test, y_train, y_test, feature_cols], shuffled_resutl


def faz_two(input_list):
    print("                                faze two                             ")

    X_train, X_test, y_train, y_true, feature_cols = input_list

    # Train Decision Tree Classifer
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy: ", accuracy)

    print("confusion matrix: \n", metrics.confusion_matrix(y_true, y_pred))

    print("precision: ", metrics.precision_score(y_true, y_pred))

    print("recall: ", metrics.recall_score(y_true, y_pred))

    print("F1: ", metrics.f1_score(y_true, y_pred))

    print("_______________________________________________________________________________")

    return accuracy


def faz_three(input_list):
    print("                                faze three                             ")

    X_train, X_test, y_train, y_true, feature_cols = input_list

    # Train Decision Tree Classifer
    clf = DecisionTreeClassifier(criterion="entropy")

    # Train Decision Tree Classifer
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy: ", metrics.accuracy_score(y_true, y_pred))

    print("confusion matrix: \n", metrics.confusion_matrix(y_true, y_pred))

    print("precision: ", metrics.precision_score(y_true, y_pred))

    print("recall: ", metrics.recall_score(y_true, y_pred))

    print("F1: ", metrics.f1_score(y_true, y_pred))

    print("_________________________________________________________________________________")

    return accuracy


def faz_four(input_list):
    print("                                faze four                             ")

    kf = KFold(n_splits=10, shuffle=False, random_state=None)
    lable_name = list(input_list.columns)
    feature_cols = lable_name[:-1]
    X = input_list[feature_cols]  # Features
    y = input_list.real  # Target variable

    best_acuracy = 0
    best_depth = 0
    list_of_depth = []
    list_of_acurrcy = []
    for val in range(5, 21):
        score = cross_val_score(DecisionTreeClassifier(max_depth=val, random_state=None), X, y, cv=kf,
                                scoring="accuracy")

        list_of_depth.append(str(val))
        list_of_acurrcy.append(score.mean())
        if score.mean() > best_acuracy:
            best_acuracy = score.mean()
            best_depth = val

    print("best depth :" + str(best_depth))
    print("it's acuracy :" + str(best_acuracy))
    print("_____________________________________________________________________")

    make_ser = pd.Series(list_of_acurrcy, index=list_of_depth)
    sns.barplot(x=make_ser, y=make_ser.index)
    # Add labels to your graph
    plt.xlabel('Tree depth')
    plt.ylabel('Accuracy')
    plt.title("Accuracy per decision tree depth on training data")
    plt.legend()
    plt.show()

    return best_acuracy, best_depth


def faz_final(shuffled_resutl):
    lable_name = list(shuffled_resutl.columns)
    feature_cols = lable_name[:-1]

    # split dataset in features and target variable
    x = shuffled_resutl[feature_cols]  # Features
    y = shuffled_resutl.real  # Target variable

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_true = train_test_split(x, y, test_size=0.2,
                                                        random_state=1)  # 80% training and 20% test

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accurecy = metrics.accuracy_score(y_true, y_pred)
    # print("Accuracy:",accurecy )

    feature_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()

    new_feature_cols = []
    number = 0

    feature_imp_dict = feature_imp.to_dict()

    for item in feature_imp_dict:

        if feature_imp_dict[item] >= 0.009:
            new_feature_cols.append(item)

    x = shuffled_resutl[new_feature_cols]  # Features

    X_train, X_test, y_train, y_true = train_test_split(x, y, test_size=0.2,
                                                        random_state=1)  # 80% training and 20% test

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accurecy2 = metrics.accuracy_score(y_true, y_pred)
    # print("new Accuracy:", accurecy2)

    return max(accurecy, accurecy2)


def main():
    result_of_faz, list_of_combine_two_csv = faz_one()

    accuracy_1 = faz_two(result_of_faz)
    #
    accuracy_2 = faz_three(result_of_faz)

    accuracy_3, depth = faz_four(list_of_combine_two_csv)

    acuracy_4 = faz_final(list_of_combine_two_csv)

    # compare the accuracy of Random Forest model with the accuracy of three former tasks

    acuracy_series = pd.Series([accuracy_1, accuracy_2, accuracy_3, acuracy_4],
                               index=["decision tree Gini", "decision tree gain ",
                                      "10-Fold with best depth " + str(depth), "Random Forest"])

    sns.barplot(x=acuracy_series, y=acuracy_series.index)
    # Add labels to your graph
    plt.xlabel('Faze of project ')
    plt.ylabel('name of Faze')
    plt.title("compare the accuracy of Random Forest model with the accuracy of three former tasks")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

