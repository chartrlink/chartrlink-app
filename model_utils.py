
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Example training function
def train_empty_leg_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    le_aircraft = LabelEncoder()
    le_operator = LabelEncoder()
    le_origin = LabelEncoder()
    le_destination = LabelEncoder()
    le_base = LabelEncoder()

    df['aircraft_type_enc'] = le_aircraft.fit_transform(df['aircraft_type'])
    df['operator_enc'] = le_operator.fit_transform(df['operator'])
    df['origin_enc'] = le_origin.fit_transform(df['origin'])
    df['destination_enc'] = le_destination.fit_transform(df['destination'])
    df['base_enc'] = le_base.fit_transform(df['aircraft_base'])

    X = df[['aircraft_type_enc', 'operator_enc', 'origin_enc', 'destination_enc', 'base_enc']]
    y = df['is_one_way']

    if len(set(y)) < 2:
        raise ValueError("Training data must include both 0s and 1s in 'is_one_way' column.")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    encoders = {
        'aircraft': le_aircraft,
        'operator': le_operator,
        'origin': le_origin,
        'destination': le_destination,
        'base': le_base
    }

    return clf, encoders
# Example prediction function
def predict_empty_leg(df, model, encoders):
    df['aircraft_type_enc'] = encoders['aircraft'].transform(df['aircraft_type'])
    df['operator_enc'] = encoders['operator'].transform(df['operator'])
    df['origin_enc'] = encoders['origin'].transform(df['origin'])
    df['destination_enc'] = encoders['destination'].transform(df['destination'])
    df['base_enc'] = encoders['base'].transform(df['aircraft_base'])

    X = df[['aircraft_type_enc', 'operator_enc', 'origin_enc', 'destination_enc', 'base_enc']]
    probas = model.predict_proba(X)[:, 1]
    return probas
