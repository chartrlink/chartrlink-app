
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_operator_data():
    return pd.read_csv("FAA_LIST_FILTERED.csv")

def main():
    st.set_page_config(page_title="Charter Flight Prediction Tool", layout="wide")
    st.title("🛩️ Charter Flight Intelligence Platform")

    tab1, tab2, tab3, tab4 = st.tabs(["Predictive Engine", "Operator Inventory", "Lead Targeting", "Dashboard Insights"])

    # TAB 1: Prediction Engine
    with tab1:
        st.subheader("🔍 Predict Empty Legs and Repositioning")
        st.markdown("Upload your upcoming charter schedule or simulate data.")

        uploaded_file = st.file_uploader("Upload CSV with flight data", type=["csv"])
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.dataframe(df_input.head())
            if 'is_one_way' in df_input.columns:
                from model_utils import train_empty_leg_model, predict_empty_leg
                model, encoders = train_empty_leg_model(df_input)
                df_input['empty_leg_proba'] = predict_empty_leg(df_input, model, encoders)

                st.success("✅ Prediction complete.")
                st.dataframe(df_input[['origin', 'destination', 'aircraft_type', 'empty_leg_proba']])
                st.download_button("📥 Download CSV with Predictions", df_input.to_csv(index=False), "empty_leg_predictions.csv")
            else:
                st.warning("⚠️ Your file must include a column called 'is_one_way' with 0 or 1 values for training.")
        else:
            st.info("No file uploaded. Please upload a charter flight schedule.")

    # TAB 2: Operator Aircraft Viewer
    with tab2:
        st.subheader("🧾 Part 135 Operator Inventory")
        df_ops = load_operator_data()

        st.sidebar.header("Filter Operators")
        min_aircraft = st.sidebar.slider("Minimum # of Aircraft", 1, 20, 3)
        manufacturer_filter = st.sidebar.multiselect(
            "Select Manufacturer(s)", df_ops['Manufacturer'].unique(), default=df_ops['Manufacturer'].unique()
        )

        filtered_ops = df_ops[df_ops['Manufacturer'].isin(manufacturer_filter)]
        operator_counts = filtered_ops.groupby('Part 135 Certificate Holder Name').size().reset_index(name='Aircraft Count')
        operator_counts = operator_counts[operator_counts['Aircraft Count'] >= min_aircraft]

        st.dataframe(operator_counts.sort_values(by='Aircraft Count', ascending=False))

    # TAB 3: Lead Targeting Tool
    with tab3:
        st.subheader("🎯 Target High-Value Operators")
        st.markdown("Use this tool to export lead lists by fleet profile")

        group = df_ops.groupby(['Manufacturer', 'Part 135 Certificate Holder Name']).size().reset_index(name='Count')
        mfr_select = st.selectbox("Select Aircraft Brand", df_ops['Manufacturer'].unique())
        count_threshold = st.slider("Minimum Fleet Count", 1, 10, 3)

        leads = group[(group['Manufacturer'] == mfr_select) & (group['Count'] >= count_threshold)]
        st.dataframe(leads)
        st.download_button("Download Leads as CSV", leads.to_csv(index=False), file_name="leads.csv")

    # TAB 4: Dashboard Insights
    with tab4:
        st.subheader("📊 Visual Intelligence Dashboard")

        if 'df_input' in locals() and 'empty_leg_proba' in df_input.columns:
            st.markdown("### ✈️ Top Empty Leg Routes")
            high_conf = df_input[df_input['empty_leg_proba'] > 0.7]
            route_counts = high_conf.groupby(['origin', 'destination']).size().reset_index(name='count')
            top_routes = route_counts.sort_values('count', ascending=False).head(10)
            st.dataframe(top_routes)

            st.markdown("### 🛩️ Aircraft Types Flying One-Way Most Often")
            top_aircraft = high_conf['aircraft_type'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_aircraft.values, y=top_aircraft.index, ax=ax)
            ax.set_xlabel("# Flights")
            ax.set_ylabel("Aircraft Type")
            st.pyplot(fig)

            st.markdown("### 📍 Most Common Empty Leg Origins")
            top_origins = high_conf['origin'].value_counts().head(10)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=top_origins.values, y=top_origins.index, ax=ax2)
            ax2.set_xlabel("# Flights")
            ax2.set_ylabel("Origin Airport")
            st.pyplot(fig2)
        else:
            st.info("📁 Upload flight data and run predictions in Tab 1 to unlock dashboards.")

if __name__ == "__main__":
    main()
