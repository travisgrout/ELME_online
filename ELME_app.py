# ELME_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_FOLDER = "Cleaned Census Inputs"

# --- Data Loading and Processing Functions (Cached for performance) ---

@st.cache_data
def load_selection_data():
    """Loads data from cleaned_zip_totals.csv to populate selection menus."""
    totals_path = os.path.join(DATA_FOLDER, "cleaned_zip_totals.csv")
    try:
        return pd.read_csv(totals_path, dtype={'zip': str})
    except FileNotFoundError:
        st.error(f"Error: `cleaned_zip_totals.csv` not found in '{DATA_FOLDER}'.")
        st.stop()

@st.cache_data
def load_naics_data():
    """Loads NAICS code information from the NAICS_to_ELME.csv file."""
    naics_path = "NAICS_to_ELME.csv"
    try:
        # Load NAICS codes as strings to preserve any leading zeros
        return pd.read_csv(naics_path, dtype={'naics': str})
    except FileNotFoundError:
        st.error(f"Error: `{naics_path}` not found. This file is required for industry selection.")
        st.stop()

@st.cache_data
def load_state_data(selected_states):
    """Dynamically loads and combines state-specific industry CSVs."""
    state_dfs = []
    for state in selected_states:
        state_file = f"cleaned_zip_{state.replace(' ', '')}.csv"
        state_path = os.path.join(DATA_FOLDER, state_file)
        try:
            state_dfs.append(pd.read_csv(state_path, dtype={'zip': str, 'fips': str, 'naics': str}))
        except FileNotFoundError:
            st.warning(f"Warning: Data file not found for {state}. Skipping.")
    if not state_dfs:
        st.error("No data could be loaded for the selected states.")
        st.stop()
    return pd.concat(state_dfs, ignore_index=True)

@st.cache_data
def calculate_employment(_df):
    """Directly calculates estimated employment from the provided dataframe."""
    df_processed = _df.copy()
    df_processed['estimated_employment'] = 0.0
    midpoints = {
        'n1_4': 2.5, 'n5_9': 7.0, 'n10_19': 14.5, 'n20_49': 34.5, 'n50_99': 74.5,
        'n100_249': 174.5, 'n250_499': 374.5, 'n500_999': 749.5, 'n1000': 1500.0
    }
    for col, midpoint in midpoints.items():
        if col in df_processed.columns:
            numeric_col = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            df_processed['estimated_employment'] += numeric_col * midpoint
    return df_processed

# --- App Initialization ---

st.set_page_config(layout="wide")
st.title("🚢 Marine Economy Employment Estimator")

# Initialize session state to manage the user's progress
if 'step' not in st.session_state:
    st.session_state.step = 'state_selection'
    st.session_state.selections = {}

# Load data for populating the menus
selection_df = load_selection_data()
naics_df = load_naics_data()

# --- Step 1: State Selection ---
if st.session_state.step == 'state_selection':
    st.header("Step 1: Select State(s)")
    states = sorted(selection_df['state'].dropna().unique())
    state_df = pd.DataFrame({'State': states})
    state_df['Select'] = False
    edited_df = st.data_editor(state_df[['Select', 'State']], hide_index=True, use_container_width=True)
    
    if st.button("Next: Select Counties", type="primary"):
        st.session_state.selections['states'] = edited_df[edited_df['Select']]['State'].tolist()
        if not st.session_state.selections['states']:
            st.warning("Please select at least one state.")
        else:
            st.session_state.step = 'county_selection'
            st.rerun()

# --- Step 2: County Selection ---
elif st.session_state.step == 'county_selection':
    st.header("Step 2: Select County(ies)")
    filtered_df = selection_df[selection_df['state'].isin(st.session_state.selections['states'])]
    counties = sorted(filtered_df['cty_name'].dropna().unique())
    county_df = pd.DataFrame({'County': counties})
    county_df['Select'] = False
    edited_df = st.data_editor(county_df[['Select', 'County']], hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to State Selection"):
            st.session_state.step = 'state_selection'
            st.rerun()
    with col2:
        if st.button("Next: Select ZIP Codes", type="primary"):
            st.session_state.selections['counties'] = edited_df[edited_df['Select']]['County'].tolist()
            if not st.session_state.selections['counties']:
                st.warning("Please select at least one county.")
            else:
                st.session_state.step = 'zip_selection'
                st.rerun()

# --- Step 3: ZIP Code Selection ---
elif st.session_state.step == 'zip_selection':
    st.header("Step 3: Select ZIP Code(s)")
    st.info("Coastal ZIP codes are selected by default.")
    
    filtered_df = selection_df[selection_df['cty_name'].isin(st.session_state.selections['counties'])]
    zip_df = filtered_df[['zip', 'city', 'cty_name', 'coastalZip']].copy().dropna().drop_duplicates()
    zip_df.rename(columns={'zip': 'ZIP', 'city': 'City', 'cty_name': 'County', 'coastalZip': 'Coastal'}, inplace=True)
    zip_df['Coastal'] = zip_df['Coastal'].apply(lambda x: 'Yes' if x == 1 else 'No')
    zip_df['Select'] = zip_df['Coastal'] == 'Yes'
    zip_df_sorted = zip_df.sort_values(by=['Select', 'ZIP'], ascending=[False, True])
    edited_df = st.data_editor(zip_df_sorted[['Select', 'ZIP', 'City', 'County', 'Coastal']], hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to County Selection"):
            st.session_state.step = 'county_selection'
            st.rerun()
    with col2:
        if st.button("Next: Select Industries", type="primary"):
            st.session_state.selections['zips'] = edited_df[edited_df['Select']]['ZIP'].tolist()
            if not st.session_state.selections['zips']:
                st.warning("Please select at least one ZIP code.")
            else:
                st.session_state.step = 'naics_selection'
                st.rerun()
                
# --- Step 4: NAICS Selection (Now using external CSV) ---
elif st.session_state.step == 'naics_selection':
    st.header("Step 4: Select Industries")

    # Prepare the dataframe from the loaded NAICS csv
    naics_selection_df = naics_df[['naics', 'Description', 'ENOW Sector']].copy()
    naics_selection_df['Select'] = True # Default to all selected
    naics_selection_df.rename(columns={'naics': 'NAICS Code'}, inplace=True)

    edited_df = st.data_editor(
        naics_selection_df[['Select', 'NAICS Code', 'Description', 'ENOW Sector']],
        hide_index=True,
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to ZIP Code Selection"):
            st.session_state.step = 'zip_selection'
            st.rerun()
    with col2:
        if st.button("Generate Employment Estimates", type="primary"):
            st.session_state.selections['naics'] = edited_df[edited_df['Select']]['NAICS Code'].tolist()
            if not st.session_state.selections['naics']:
                st.warning("Please select at least one industry.")
            else:
                st.session_state.step = 'show_results'
                st.rerun()

# --- Step 5: Show Results ---
elif st.session_state.step == 'show_results':
    st.header("Final Employment Estimates")
    
    selections = st.session_state.selections
    st.write("Based on your selections:")
    st.write(f"**States:** {', '.join(selections['states'])}")
    st.write(f"**Counties:** {len(selections['counties'])} selected")
    st.write(f"**ZIP Codes:** {len(selections['zips'])} selected")
    st.write(f"**Industries:** {len(selections['naics'])} selected")
    
    with st.spinner('Loading data and calculating...'):
        state_level_data = load_state_data(selections['states'])
        processed_data = calculate_employment(state_level_data)
        
        mask = (
            processed_data['zip'].isin(selections['zips']) &
            processed_data['naics'].isin(selections['naics'])
        )
        final_df = processed_data[mask]

        if final_df.empty:
            st.info("No data available for the selected criteria.")
        else:
            summary = final_df.groupby('naics')['estimated_employment'].sum().reset_index()
            
            # Merge with the NAICS data to get descriptions and sectors
            # Ensure the 'naics' column is used for merging
            naics_info = naics_df[['naics', 'Description', 'ENOW Sector']]
            summary = pd.merge(summary, naics_info, on='naics', how='left')
            
            summary.rename(columns={
                'naics': 'NAICS Code',
                'estimated_employment': 'Estimated Employment'
            }, inplace=True)
            
            st.dataframe(
                summary[['NAICS Code', 'Description', 'ENOW Sector', 'Estimated Employment']].style.format({'Estimated Employment': '{:,.0f}'}),
                use_container_width=True
            )
            total_employment = summary['Estimated Employment'].sum()
            st.metric(label="Total Estimated Employment for Selection", value=f"{total_employment:,.0f}")
    
    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
