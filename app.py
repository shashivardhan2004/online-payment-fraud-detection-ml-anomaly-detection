
import os
import json
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Online Payment Fraud Detection",
    layout="wide"
)


if not os.path.exists(
    "fraud_model.joblib"
):

    st.error(
        "The saved model file is missing."
    )

    st.stop()


if not os.path.exists(
    "model_metadata.json"
):

    st.error(
        "The model metadata file is missing."
    )

    st.stop()


model = joblib.load(
    "fraud_model.joblib"
)


with open(
    "model_metadata.json",
    "r"
) as file:

    metadata = json.load(
        file
    )


threshold = float(
    metadata[
        "threshold"
    ]
)


st.title(
    "Online Payment Fraud Detection"
)


st.caption(
    "Machine Learning and Anomaly Detection Research Prototype"
)


if metadata.get(
    "run_mode"
) != "FINAL":

    st.warning(
        "This application was produced in CHECK mode. "
        "These results are not my final dissertation results."
    )


st.info(
    "This is a research prototype created using simulated Sparkov data. "
    "It is not intended for live banking or automatic transaction blocking."
)


tab1, tab2, tab3, tab4 = st.tabs([
    "Fraud Prediction",
    "Model Performance",
    "Explainability",
    "About"
])


with tab1:

    st.subheader(
        "Transaction Risk Assessment"
    )


    left, right = st.columns(
        2
    )


    with left:

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )


        city_population = st.number_input(
            "Customer city population",
            min_value=1,
            value=50000,
            step=100
        )


        age = st.number_input(
            "Customer age",
            min_value=18.0,
            max_value=120.0,
            value=35.0,
            step=1.0
        )


        distance = st.number_input(
            "Customer-to-merchant distance (km)",
            min_value=0.0,
            value=25.0,
            step=1.0
        )


        category = st.selectbox(
            "Merchant category",
            metadata[
                "category_values"
            ]
        )


    with right:

        gender = st.selectbox(
            "Gender",
            metadata[
                "gender_values"
            ]
        )


        state = st.selectbox(
            "State",
            metadata[
                "state_values"
            ]
        )


        hour = st.slider(
            "Hour",
            0,
            23,
            12
        )


        weekday = st.selectbox(
            "Day of week",
            list(
                range(
                    7
                )
            ),
            format_func=lambda value: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][value]
        )


        month = st.slider(
            "Month",
            1,
            12,
            6
        )


    if st.button(
        "Assess Fraud Risk"
    ):

        input_data = pd.DataFrame([
            {
                "amt":
                    amount,

                "city_pop":
                    city_population,

                "hour":
                    hour,

                "weekday":
                    weekday,

                "month":
                    month,

                "is_weekend":
                    int(
                        weekday
                        >= 5
                    ),

                "age":
                    age,

                "distance_km":
                    distance,

                "category":
                    category,

                "gender":
                    gender,

                "state":
                    state
            }
        ])


        risk_score = float(
            model
            .predict_proba(
                input_data
            )[
                0,
                1
            ]
        )


        prediction = int(
            risk_score
            >=
            threshold
        )


        result_col1, result_col2 = (
            st.columns(
                2
            )
        )


        result_col1.metric(
            "Fraud risk score",
            f"{risk_score:.2%}"
        )


        result_col2.metric(
            "Decision threshold",
            f"{threshold:.4f}"
        )


        if prediction == 1:

            st.error(
                "HIGH FRAUD RISK"
            )


        else:

            st.success(
                "LOW FRAUD RISK"
            )


        st.caption(
            "The risk score is a model output and is not claimed "
            "to be a calibrated real-world banking probability."
        )


with tab2:

    st.subheader(
        "Hold-Out Model Performance"
    )


    metrics = metadata[
        "metrics"
    ]


    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )


    col1.metric(
        "Precision",
        f"{metrics.get('precision', 0):.4f}"
    )


    col2.metric(
        "Recall",
        f"{metrics.get('recall', 0):.4f}"
    )


    col3.metric(
        "F1",
        f"{metrics.get('f1', 0):.4f}"
    )


    col4.metric(
        "PR-AUC",
        f"{metrics.get('PR_AUC', 0):.4f}"
    )


    st.write(
        "**ROC-AUC:**",
        metrics.get(
            "ROC_AUC"
        )
    )


    st.write(
        "**True positives:**",
        metrics.get(
            "TP"
        )
    )


    st.write(
        "**False positives:**",
        metrics.get(
            "FP"
        )
    )


    st.write(
        "**False negatives:**",
        metrics.get(
            "FN"
        )
    )


with tab3:

    st.subheader(
        "Model Explainability"
    )


    st.write(
        "Feature importance shows which inputs have the strongest "
        "influence on the model. It does not prove causation."
    )


    if os.path.exists(
        "shap_feature_importance.csv"
    ):

        importance = pd.read_csv(
            "shap_feature_importance.csv"
        )


        st.dataframe(
            importance.head(
                20
            ),
            use_container_width=True
        )


        st.bar_chart(
            importance
            .head(
                15
            )
            .set_index(
                "feature"
            )[
                "importance"
            ]
        )


    elif os.path.exists(
        "model_feature_importance.csv"
    ):

        importance = pd.read_csv(
            "model_feature_importance.csv"
        )


        st.dataframe(
            importance.head(
                20
            ),
            use_container_width=True
        )


    else:

        st.info(
            "Explainability output is not available."
        )


with tab4:

    st.subheader(
        "About the Project"
    )


    st.write(
        f"""
        **Selected model:** {metadata['model_name']}

        **Imbalance strategy:** {metadata['imbalance_strategy']}

        **Decision threshold:** {metadata['threshold']:.4f}

        **Run mode:** {metadata['run_mode']}

        This project compares Logistic Regression,
        Random Forest, XGBoost and Isolation Forest
        for fraud detection under severe class imbalance.

        The training experiments compare the original class
        distribution, class weighting, random undersampling and SMOTE.

        This application is a research prototype and should not
        be used to automatically decline real financial transactions.
        """
    )
