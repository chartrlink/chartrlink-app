from model_utils import train_empty_leg_model, predict_empty_leg
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_operator_data():
    return pd.read_csv("FAA_LIST_FILTERED.csv")

def main():
    st.set_page_config(page_title="Charter Flight Prediction Tool", layout="wide")
    st.title("🛩️ Charter Flight Intelligence Platform")

    tab1, tab2, tab3 = st.tabs(["Predictive Engine", "Operator Inventory", "Lead Targeting"])

    # TAB 1: Prediction Engine
    with tab1:
        st.subheader("🔍 Predict Empty Legs and Repositioning")
        st.markdown("Upload your upcoming charter schedule or simulate data.")

        uploaded_file = st.file_uploader("Upload CSV with flight data", type=["csv"])
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.dataframe(df_input.head())
            st.success("Prediction feature coming soon: confidence score + next destination")
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

if __name__ == "__main__":
    main()
