
from flask import Flask, render_template, request,flash
import pandas as pd
from flask import session
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from sklearn.model_selection import train_test_split
from DBConfig import DBConnection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from RF import rfc_evaluation
from Adaboost import ab_evaluation
from ETC import etc_evaluation
from VTC import vtc_evaluation
import numpy as np



app = Flask(__name__)
app.secret_key = "abc"


dict={}

accuracy_list=[]
accuracy_list.clear()
precision_list=[]
precision_list.clear()
recall_list=[]
recall_list.clear()
f1score_list=[]
f1score_list.clear()



@app.route('/')
def index():
    return render_template('index.html')


@app.route("/user")
def user():
    return render_template("user.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/fooddiet")
def fooddiet():
    return render_template("fooddiet.html")


@app.route("/newuser")
def newuser():
    return render_template("register.html")


@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")


@app.route("/preprocessing")
def preprocessing():
    return render_template("data_preprocessing.html")


@app.route("/perevaluations")
def perevaluations():
    accuracy_graph()
    precision_graph()
    recall_graph()
    f1score_graph()

    return render_template("metrics.html")




@app.route("/data_preprocessing" ,methods =["GET", "POST"] )
def data_preprocessing():
    fname = request.form.get("file")
    df = pd.read_csv("../GDM/dataset/"+fname)
    df1 = df.dropna()
    y= df1['ClassLabel']
    del df1['ClassLabel']
    X=df1
    #print(X)
    dict['X'] = X
    dict['y'] = y

    return render_template("data_preprocessing.html",msg="Data Preprocessing Completed..!")


@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        uid = request.form.get('unm')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from physicians where userid='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into physicians values(%s,%s,%s,%s,%s)"
            values = (name,uid, pwd,email,mno)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user.html", msg="Registered Successfully..! Login Here.")


        else:
            return render_template("register.html", msg="User name already exists..!")



    except Exception as e:
        print(e)

    return ""

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from physicians where userid='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html")
        else:

            return render_template("user.html", msg2="Invalid Credentials")

        return ""



@app.route("/evaluations" )
def evaluations():
        rf_list = []
        etc_list = []
        abc_list = []
        vtc_list = []
        metrics = []

        X = dict['X']

        y = dict['y']

        # Split train test: 70 % - 30 %
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        accuracy_abc, precision_abc, recall_abc, fscore_abc = ab_evaluation(X_train, X_test, y_train, y_test)
        abc_list.append("AdaboostClassifier")
        abc_list.append(accuracy_abc)
        abc_list.append(precision_abc)
        abc_list.append(recall_abc)
        abc_list.append(fscore_abc)

        accuracy_etc, precision_etc, recall_etc, fscore_etc = etc_evaluation(X_train, X_test, y_train, y_test)
        etc_list.append("ExtraTreesClassifier")
        etc_list.append(accuracy_etc)
        etc_list.append(precision_etc)
        etc_list.append(recall_etc)
        etc_list.append(fscore_etc)

        accuracy_rf, precision_rf, recall_rf, fscore_rf = rfc_evaluation(X_train, X_test, y_train, y_test)
        rf_list.append("RandomForest")
        rf_list.append(accuracy_rf)
        rf_list.append(precision_rf)
        rf_list.append(recall_rf)
        rf_list.append(fscore_rf)

        abc_clf = AdaBoostClassifier()
        etc_clf = ExtraTreesClassifier()
        rfc_clf = RandomForestClassifier()
        accuracy_vtc, precision_vtc, recall_vtc, fscore_vtc = vtc_evaluation(X_train, X_test, y_train, y_test, abc_clf,
                                                                             etc_clf, rfc_clf)
        vtc_list.append("VTC")
        vtc_list.append(accuracy_vtc)
        vtc_list.append(precision_vtc)
        vtc_list.append(recall_vtc)
        vtc_list.append(fscore_vtc)

        metrics.clear()
        metrics.append(abc_list)
        metrics.append(etc_list)
        metrics.append(rf_list)
        metrics.append(vtc_list)

        return render_template("evaluations.html", evaluations=metrics)





def accuracy_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    accuracy_list.clear()

    cursor.execute("select accuracy from evaluations")
    acdata=cursor.fetchall()

    for record in acdata:
        accuracy_list.append(float(record[0]))

    height = accuracy_list
    print("height=",height)
    bars = ('ABC','ETC','RF','VTC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['red', 'green', 'blue', 'orange'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Analysis on ML Accuracies')
    plt.savefig('static/accuracy.png')
    plt.clf()
    #plt.savefig('accuracy.png')

    return ""


def precision_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select precesion from evaluations")
    pdata = cursor.fetchall()

    precision_list.clear()
    for record in pdata:
        precision_list.append(float(record[0]))

    height = precision_list
    print("pheight=",height)
    bars = ('ABC','ETC','RF','VTC')
    y_pos = np.arange(len(bars))
    plt2.bar(y_pos, height, color=['green', 'brown', 'violet', 'blue'])
    plt2.xticks(y_pos, bars)
    plt2.xlabel('Algorithms')
    plt2.ylabel('Precision')
    plt2.title('Analysis on ML Precisions')
    plt2.savefig('static/precision.png')
    plt2.clf()
    return ""

def recall_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    recall_list.clear()
    cursor.execute("select recall from evaluations")
    recdata = cursor.fetchall()

    for record in recdata:
        recall_list.append(float(record[0]))

    height = recall_list

    bars = ('ABC','ETC','RF','VTC')
    y_pos = np.arange(len(bars))
    plt3.bar(y_pos, height, color=['orange', 'cyan', 'gray', 'violet'])
    plt3.xticks(y_pos, bars)
    plt3.xlabel('Algorithms')
    plt3.ylabel('Recall')
    plt3.title('Analysis on ML Recall')
    plt3.savefig('static/recall.png')
    plt3.clf()
    return ""


def f1score_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    f1score_list.clear()

    cursor.execute("select f1score from evaluations")
    fsdata = cursor.fetchall()

    for record in fsdata:
        f1score_list.append(float(record[0]))

    height = f1score_list
    print("fheight=",height)
    bars = ('ABC','ETC','RF','VTC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['gray', 'green', 'orange', 'brown'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('F1-Score')
    plt.title('Analysis on ML F1-Score')
    plt4.savefig('static/f1score.png')

    return ""


@app.route("/gdm_prediction")
def gdm_prediction():
    return render_template("gdm_prediction.html")


@app.route("/prediction", methods =["GET", "POST"])
def prediction():
    age = request.form.get("age")
    npg = request.form.get("npg")
    gpp = request.form.get("gpp")
    bmi = request.form.get("bmi")
    hdl = request.form.get("hdl")
    fhd = request.form.get("fhd")
    upl = request.form.get("upl")
    bd = request.form.get("bd")
    pcos = request.form.get("pcos")

    sysbp = request.form.get("sysbp")
    disbp = request.form.get("disbp")
    ogtt = request.form.get("ogtt")
    hmgbn = request.form.get("hmgbn")
    sl = request.form.get("sl")
    pdb = request.form.get("pd")



    df = pd.read_csv("../GDM/dataset/GDM.csv" )
    df1 = df.dropna()

    y_train = df1['ClassLabel']
    del df1['ClassLabel']
    x_train = df1


    X_test=[[float(age),float(npg),float(gpp),float(bmi),float(hdl),float(fhd),float(upl),float(bd),float(pcos),
             float(sysbp),float(disbp),float(ogtt),float(hmgbn),float(sl),float(pdb)]]

    abc_clf = AdaBoostClassifier()
    etc_clf = ExtraTreesClassifier()
    rfc_clf = RandomForestClassifier()


    voting_clf = VotingClassifier(
        estimators=[('RF', rfc_clf), ('abc_clf', abc_clf), ('etc_clf', etc_clf)],
        voting='hard')

    voting_clf.fit(x_train, y_train)
    predicted = voting_clf.predict(np.array(X_test))
    result = predicted[0]
    print(result)

    if result==1:
        result="POSITIVE"
    else:
        result = "NEGATIVE"

    print("res=", result)


    return render_template("gdm_prediction.html",result=result)

if __name__ == '__main__':
    app.run(host="localhost", port=2468, debug=True)
