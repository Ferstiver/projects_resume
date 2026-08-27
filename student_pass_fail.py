import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("students_messy.csv")

#print(df.shape)
#print(df.dtypes)
#print(df.info())
#print(df.isna().sum())
#print("\n",df["course"].unique())
#print("\n",df["result"].unique())
#print("\n",df["hours"].unique()[:30])

df["hours"] = pd.to_numeric(df["hours"], errors= "coerce")
df["attendance"] = pd.to_numeric(df["attendance"], errors= "coerce")
df["previous_score"] = pd.to_numeric(df["previous_score"], errors= "coerce")

df["course"] = df["course"].replace("N/A", pd.NA)
df["result"] = df["result"].map({"Fail":0, "Pass": 1})

df = df.dropna()
print(df.shape)
print(df.dtypes)
#print(df["result"].unique())
#print(df["course"].unique())

df_encoded = pd.get_dummies(df,columns=["course"], dtype= int)
X = df_encoded.drop(columns=["name", "result"])
y = df_encoded["result"]

model = LogisticRegression(max_iter=1000) 
model_1 = RandomForestClassifier(random_state=42)

lr_score = cross_val_score(model,X,y, cv=5 ,scoring="accuracy")
rf_score = cross_val_score(model_1,X,y, cv=5, scoring="accuracy")

print("Logistic Regression avg:", round(lr_score.mean(),2))
print("Random Forest Classifier avg:", round(rf_score.mean(),2))

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify= y
)


model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nPredictions:", [round(x) for x in predictions])
print("\nAccuracy:", round(accuracy_score(y_test, predictions),2))
print("\nConfusion matrix:", confusion_matrix(y_test, predictions))
print("\nClassification report:", classification_report(y_test, predictions))

new_student = pd.DataFrame([{
    "hours" : 8.5,
    "attendance" : 82,
    "previous_score" : 74,
    "course" : "CS"
}])

new_student_en = pd.get_dummies(new_student, columns=["course"], dtype=int)
new_student_en = new_student_en.reindex(columns=X.columns, fill_value=0)
print(model.predict(new_student_en)[0])
