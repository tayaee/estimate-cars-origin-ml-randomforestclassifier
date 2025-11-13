import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Car Origin Prediction", layout="wide")

st.title("Car Origin Prediction")
st.markdown("Random Forest Classifier prediction. Adjust parameters for instant results.")


@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("data/cars.csv")
    except FileNotFoundError:
        st.error("Data file not found.")
        return None, None, None, None, None, None

    df = df.dropna()

    feature_names = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
    ]
    X = df[feature_names]
    y = df["origin"]

    X_encoded = pd.get_dummies(X, columns=["model year"], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test, X_encoded.columns, y.unique()


y_test: pd.Series
X_train, X_test, y_train, y_test, feature_names, class_labels = load_and_preprocess_data()  # type: ignore

if X_train is None:
    st.stop()


def train_and_predict(
    X_train,
    X_test,
    y_train,
    y_test,
    max_features,
    min_samples_leaf,
    min_samples_split,
    class_weight_param,
):
    rforest = RandomForestClassifier(
        n_estimators=100,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight_param,
        random_state=1,
        n_jobs=-1,
    )
    rforest.fit(X_train, y_train)

    y_pred_test = rforest.predict(X_test)

    return y_pred_test, rforest


st.sidebar.header("Model Hyperparameters")

max_features_options = list(range(1, min(10, X_train.shape[1] + 1)))
selected_max_features = st.sidebar.select_slider("max_features", options=max_features_options, value=3)
selected_min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=20, value=4)
selected_min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=20, value=2)
selected_class_weight = st.sidebar.selectbox(
    "class_weight",
    options=["None", "balanced", "balanced_subsample"],
    index=0,
)

class_weight_param = None if selected_class_weight == "None" else selected_class_weight

st.sidebar.markdown("---")
st.sidebar.write("Current Params:")
st.sidebar.write(f"- `max_features`: {selected_max_features}")
st.sidebar.write(f"- `min_samples_leaf`: {selected_min_samples_leaf}")
st.sidebar.write(f"- `min_samples_split`: {selected_min_samples_split}")
st.sidebar.write(f"- `class_weight`: {selected_class_weight}")

st.info("Training and evaluating model...")

y_pred_test, model = train_and_predict(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_max_features,
    selected_min_samples_leaf,
    selected_min_samples_split,
    class_weight_param,
)

st.header("Model Evaluation (Test Set)")

report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

col1, col2, col3, col4 = st.columns(4)

accuracy = accuracy_score(y_test, y_pred_test)
col1.metric("Accuracy", f"{accuracy:.4f}")

col2.metric("Precision (Macro)", f"{report['macro avg']['precision']:.4f}")
col3.metric("Recall (Macro)", f"{report['macro avg']['recall']:.4f}")
col4.metric("F1-Score (Macro)", f"{report['macro avg']['f1-score']:.4f}")

st.subheader("Feature Importance")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
    by="Importance", ascending=False
)

fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="magma", ax=ax_imp)
ax_imp.set_title("Feature Importance", fontsize=16)
ax_imp.set_xlabel("Importance", fontsize=12)
ax_imp.set_ylabel("Feature", fontsize=12)
st.pyplot(fig_imp)

st.subheader("Classification Report")
report_df = pd.DataFrame(report).transpose().round(4)
st.dataframe(report_df)
