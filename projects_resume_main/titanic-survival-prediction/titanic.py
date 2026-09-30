import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df= pd.read_csv("titanic.csv")

print(df.dtypes)
print(df.shape)
#print(df["PassengerId"].unique())
print(df["Survived"].unique())
print(df["Pclass"].unique())
print(df["Age"].unique())
print(df["Embarked"].unique())


df["Age"] = pd.to_numeric(df["Age"], errors= "coerce")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df = df.drop(columns=["Cabin"])
df["Embarked"] = df["Embarked"].fillna("S")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"]

X = df[features]
y = df["Survived"]

X_train,X_test, y_train, y_test = train_test_split(
    X,y , test_size=0.2, random_state=42
)

numerical_features = [ "Age", "FamilySize","Fare", "IsAlone"]
categorical_features = ["Pclass","Embarked", "Sex"]

preprocess = ColumnTransformer([
    ("num",StandardScaler(), numerical_features),
    ("cat",OneHotEncoder(handle_unknown= "ignore"), categorical_features)
])


pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier())
])


param = {
    "model__n_estimators" : [50,100],
    "model__max_depth" : [2,4, None]
}

grid = GridSearchCV(
    pipe,
    param,
    cv = 5,
    scoring= "accuracy"
)

grid.fit(X_train,y_train)


print("Best:", grid.best_params_)
print("Score:", grid.best_score_)


y_pred = grid.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))


test_passengers = pd.DataFrame([
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "Fare": 100,
        "Embarked": "S",
        "FamilySize": 1,
        "IsAlone": 1
    }
])

predictions = grid.predict(test_passengers)
probabilities = grid.predict_proba(test_passengers)
print(predictions)
print(probabilities)
