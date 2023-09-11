
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

from DBConfig import DBConnection

def rfc_evaluation(X_train, X_test, y_train, y_test):

    db = DBConnection.getConnection()
    cursor = db.cursor()
    rfc_clf = RandomForestClassifier()

    rfc_clf.fit(X_train, y_train)

    predicted = rfc_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100

    values= ("RF", accuracy, precision, recall, fscore)
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()
    print("RF=",accuracy,precision,recall,fscore)
    return accuracy, precision, recall, fscore




