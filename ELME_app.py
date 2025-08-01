# app.py
import streamlit as st
import pandas as pd
import numpy as np

# --- Constants and Configuration ---

# NAICS codes and their descriptions as provided by the user
NAICS_CODES = {
    '112511': 'Finfish Farming and Fish Hatcheries', '112512': 'Shellfish Farming',
    '112519': 'Other Aquaculture', '114111': 'Finfish Fishing',
    '114112': 'Shellfish Fishing', '114119': 'Other Marine Fishing',
    '424460': 'Fish and Seafood Merchant Wholesalers', '445250': 'Fish and Seafood Retailers',
    '311710': 'Seafood Product Preparation and Packaging', '237990': 'Other Heavy and Civil Engineering Construction',
    '483111': 'Deep Sea Freight Transportation', '483113': 'Coastal and Great Lakes Freight Transportation',
    '483112': 'Deep Sea Passenger Transportation', '483114': 'Coastal and Great Lakes Passenger Transportation',
    '488310': 'Port and Harbor Operations', '488320': 'Marine Cargo Handling',
    '488330': 'Navigational Services to Shipping', '488390': 'Other Support Activities for Water Transportation',
    '334511': 'Search, Detection, Navigation, Guidance, Aeronautical, and Nautical System and Instrument Manufacturing',
    '493110': 'General Warehousing and Storage', '493120': 'Refrigerated Warehousing and Storage',
    '493130': 'Farm Product Warehousing and Storage', '212321': 'Construction Sand and Gravel Mining',
    '212322': 'Industrial Sand Mining', '211120': 'Crude Petroleum Extraction',
    '211130': 'Natural Gas Extraction', '213111': 'Drilling Oil and Gas Wells',
    '213112': 'Support Activities for Oil and Gas Operations', '541360': 'Geophysical Surveying and Mapping Services',
    '336612': 'Boat Building', '336611': 'Ship Building and Repairing',
    '487990': 'Scenic and Sightseeing Transportation, Other', '532284': 'Recreational Goods Rental',
    '611620': 'Sports and Recreation Instruction', '713990': 'All Other Amusement and Recreation Industries',
    '441222': 'Boat Dealers', '722511': 'Full-Service Restaurants',
    '722513': 'Limited-Service Restaurants', '722514': 'Cafeterias, Grill Buffets, and Buffets',
    '722515': 'Snack and Nonalcoholic Beverage Bars', '721110': 'Hotels (except Casino Hotels) and Motels',
    '721191': 'Bed-and-Breakfast Inns', '713930': 'Marinas',
    '721211': 'RV (Recreational Vehicle) Parks and Campgrounds', '487210': 'Scenic and Sightseeing Transportation, Water',
    '339920': 'Sporting and Athletic Goods Manufacturing', '712130': 'Zoos and Botanical Gardens',
    '712190': 'Nature Parks and Other Similar Institutions'
}

# Establishment size columns
SIZE_COLS = [
    "n1_4", "n5_9", "n10_19", "n20_49", "n50_99",
    "n100_249", "n250_499", "n500_999", "n1000"
]

# Mid-points for each establishment size range for employment estimation
EMPLOYMENT_MIDPOINTS = {
    'n1_4': 2.5, 'n5_9': 7.0, 'n10_19': 14.5,
    'n20_49': 34.5, 'n50_99': 74.5, 'n100_249': 174.5,
    'n250_499': 374.5, 'n500_999': 749.5, 'n1000': 1500.0 # Assumed midpoint for 1000+
}

# --- Data Loading and Preparation ---

# app.py

@st.cache_data
def load_and_prepare_data():
    """
    Loads, merges, and prepares the datasets for analysis from the 'Cleaned Census Inputs' folder.
    - Loads zip-level and county-level data.
    - Combines them into a single dataframe where county-level data is
      identified with a special zip code of -99999.
    """
    try:
        # Correctly path to the CSV files inside the subfolder
        zip_df = pd.read_csv("Cleaned Census Inputs/cleaned_zip_industries.csv", dtype={'zip': str, 'fips': str, 'naics': str})
        county_df = pd.read_csv("Cleaned Census Inputs/cleaned_cbp_counties.csv", dtype={'fipstate': str, 'fipscty': str, 'naics': str})
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure the 'Cleaned Census Inputs' folder with your CSV files is in your GitHub repository.")
        st.stop()

    # Prepare county data: create FIPS code and add sentinel zip value
    county_df['fipscty'] = county_df['fipscty'].str.zfill(3)
    county_df['fips'] = county_df['fipstate'].str.zfill(2) + county_df['fipscty']
    county_df['zip'] = -99999 # Sentinel value to identify county total rows

    # Align columns between the two dataframes for a clean merge
    common_cols = ['fips', 'naics', 'est'] + SIZE_COLS
    county_df_aligned = county_df.rename(columns={'ALLCoastalCounty': 'coastalCounty'}) # Align column name
    
    # Ensure all required columns exist, adding them with null values if they don't
    for col in common_cols + ['state', 'cty_name', 'coastalCounty', 'city']:
        if col not in zip_df.columns: zip_df[col] = np.nan
        if col not in county_df_aligned.columns: county_df_aligned[col] = np.nan

    # Select and reorder columns to match before concatenation
    zip_df_aligned = zip_df[common_cols + ['state', 'cty_name', 'zip', 'city', 'coastalCounty']]
    county_df_aligned = county_df_aligned[common_cols + ['state', 'cty_name', 'zip', 'city', 'coastalCounty']]

    # Combine zip and county data into one dataframe
    combined_df = pd.concat([zip_df_aligned, county_df_aligned], ignore_index=True)

    # Convert relevant columns to numeric types for calculations
    for col in ['est'] + SIZE_COLS:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    return combined_df

# --- Core Data Processing Functions ---

def distribute_establishments(group, size_cols):
    """
    Python translation of the R script's logic to distribute establishments.
    This function is applied to each 'naics-fips' group.
    """
    county_row = group[group['zip'] == -99999]
    zip_rows = group[group['zip'] != -99999].copy()

    if county_row.empty or zip_rows.empty:
        return group # Cannot perform distribution

    county_vals = county_row[size_cols].iloc[0].fillna(0)
    zip_sums_before = zip_rows[size_cols].sum().fillna(0)
    
    # Calculate available establishments to distribute (capacity)
    capacity = county_vals - zip_sums_before
    capacity[capacity < 0] = 0

    # Iterate through zip code rows to allocate missing establishments
    for i, row in zip_rows.iterrows():
        if pd.isna(row['est']): continue

        missing_estab = row['est'] - row[size_cols].sum()
        if missing_estab > 0:
            target_cols_mask = row[size_cols].fillna(0) == 0
            target_cols = [col for col, mask in target_cols_mask.items() if mask]

            if target_cols:
                weights = county_vals[target_cols]
                total_weight = weights.sum()

                if total_weight > 0:
                    proportions = weights / total_weight
                    allocations = missing_estab * proportions
                    zip_rows.loc[i, target_cols] += allocations
                    
    return pd.concat([county_row, zip_rows], ignore_index=True)

@st.cache_data
def run_processing(_df):
    """
    Main processing pipeline: distributes establishments and estimates employment.
    The _df parameter is prefixed with an underscore to indicate it's cached.
    """
    st.write("Processing Data: Distributing establishments...")
    processed_df = _df.groupby(['naics', 'fips'], group_keys=False).apply(lambda x: distribute_establishments(x, SIZE_COLS))
    
    st.write("Processing Data: Estimating employment...")
    zip_data = processed_df[processed_df['zip'] != -99999].copy()

    zip_data['estimated_employment'] = 0.0
    for col, midpoint in EMPLOYMENT_MIDPOINTS.items():
        zip_data[col] = zip_data[col].fillna(0)
        zip_data['estimated_employment'] += zip_data[col] * midpoint
    
    st.success("Processing complete.", icon="✅")
    return zip_data

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("🚢 Marine Economy Employment Estimator")

# Load initial data
initial_data = load_and_prepare_data()

# --- Sidebar for User Selections ---
st.sidebar.header("Step 1: Select Your Region")

# State Selection
available_states = sorted(initial_data['state'].dropna().unique())
selected_states = st.sidebar.multiselect("Select State(s)", available_states)

# County Selection
if selected_states:
    county_mask = initial_data['state'].isin(selected_states)
    available_counties = sorted(initial_data[county_mask]['cty_name'].dropna().unique())
    selected_counties = st.sidebar.multiselect("Select County(s)", available_counties)

    # Zip Code Selection
    if selected_counties:
        zip_mask = (initial_data['cty_name'].isin(selected_counties)) & (initial_data['zip'] != -99999)
        zip_options_df = initial_data[zip_mask].copy().dropna(subset=['zip', 'city', 'cty_name'])
        
        zip_options_df['display'] = zip_options_df.apply(
            lambda row: f"{row['zip']} ({row['city']}) - Coastal: {'Yes' if row['coastalCounty']==1 else 'No'}",
            axis=1
        )
        
        zip_options_map = pd.Series(zip_options_df['zip'].values, index=zip_options_df['display']).to_dict()
        available_zips_display = sorted(zip_options_map.keys())
        selected_zips_display = st.sidebar.multiselect("Select ZIP Code(s)", available_zips_display)
        selected_zips = [zip_options_map[z] for z in selected_zips_display]

# --- Industry Selection ---
st.sidebar.header("Step 2: Select Industries")
naics_display_list = [f"{code} - {desc}" for code, desc in NAICS_CODES.items()]

selected_naics_display = st.sidebar.multiselect(
    "Select from default NAICS codes:",
    options=naics_display_list,
    default=naics_display_list
)
selected_naics_codes = [item.split(' - ')[0] for item in selected_naics_display]

custom_naics_input = st.sidebar.text_area("Add custom NAICS codes (comma-separated):")
if custom_naics_input:
    custom_naics = [code.strip() for code in custom_naics_input.split(',')]
    all_selected_naics = list(set(selected_naics_codes + custom_naics))
else:
    all_selected_naics = selected_naics_codes

# --- Main Panel for Calculation and Results ---

st.header("Results")
if st.button("Generate Employment Estimates", type="primary"):
    if not selected_states or not selected_counties or not selected_zips or not all_selected_naics:
        st.warning("Please select at least one State, County, ZIP Code, and Industry.")
    else:
        with st.spinner('Calculating... This may take a moment.'):
            processed_data = run_processing(initial_data)
            
            results_mask = (processed_data['zip'].isin(selected_zips)) & (processed_data['naics'].isin(all_selected_naics))
            final_df = processed_data[results_mask]

            if final_df.empty:
                st.info("No data available for the selected criteria.")
            else:
                employment_summary = final_df.groupby('naics')['estimated_employment'].sum().reset_index()
                employment_summary['industry_description'] = employment_summary['naics'].map(NAICS_CODES).fillna("Custom or Unknown NAICS")
                
                employment_summary = employment_summary.rename(columns={
                    'naics': 'NAICS Code',
                    'estimated_employment': 'Estimated Employment',
                    'industry_description': 'Industry Description'
                })
                
                employment_summary = employment_summary[['NAICS Code', 'Industry Description', 'Estimated Employment']]
                
                st.subheader("Total Estimated Employment by Industry")
                st.dataframe(employment_summary.style.format({'Estimated Employment': '{:,.0f}'}), use_container_width=True)

                total_employment = employment_summary['Estimated Employment'].sum()
                st.metric(label="Total Estimated Employment for Selection", value=f"{total_employment:,.0f}")
