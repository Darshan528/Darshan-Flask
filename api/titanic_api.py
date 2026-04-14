from flask import Blueprint, request, jsonify
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

titanic_api = Blueprint('titanic_api', __name__)

FEATURE_NAMES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']

def _train():
    df = sns.load_dataset('titanic')
    df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']].dropna()
    df['sex']      = (df['sex'] == 'male').astype(int)
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)
    df['alone']    = df['alone'].astype(int)
    X = df[FEATURE_NAMES]
    y = df['survived']
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    importances = {f: round(float(i), 4) for f, i in zip(FEATURE_NAMES, clf.feature_importances_)}
    return clf, importances

_model, _importances = _train()


def _parse(data):
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    return [[
        int(data.get('pclass', 3)),
        1 if data.get('sex', 'male') == 'male' else 0,
        float(data.get('age', 30)),
        int(data.get('sibsp', 0)),
        int(data.get('parch', 0)),
        float(data.get('fare', 7.5)),
        embarked_map.get(data.get('embarked', 'S'), 0),
        1 if data.get('alone', True) else 0,
    ]]


@titanic_api.route('/api/titanic/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) or {}
    proba = _model.predict_proba(_parse(data))[0]
    return jsonify({'survive': round(float(proba[1]), 4), 'die': round(float(proba[0]), 4)})


@titanic_api.route('/api/titanic/feature_weights', methods=['GET'])
def feature_weights():
    return jsonify(_importances)
