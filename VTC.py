
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from DBConfig import DBConnection
def vtc_evaluation(X_train, X_test, y_train, y_test, abc_clf,etc_clf,rfc_clf):
    db = DBConnection.getConnection()
    cursor = db.cursor()
    voting_clf = VotingClassifier(estimators=[('abc', abc_clf),('etc', etc_clf),('RF', rfc_clf)], voting='hard')

    voting_clf.fit(X_train, y_train)

    predicted = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100

    values = ("VotingClassifier", accuracy, precision, recall, fscore)
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()
    print("VTC=",accuracy,precision,recall,fscore)
    return accuracy, precision, recall, fscore




