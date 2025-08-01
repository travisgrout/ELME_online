import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Marine Economy Estimator",
    page_icon="🌊",
    layout="wide"
)

# --- Caching Data Loading ---
@st.cache_data
def load_data(year="23"):
    """
    Loads all necessary RAW data from txt and CSV files.
    This function is cached to improve performance.
    """
    try:
        # Define file paths based on the year
        zbp_detail_file = f"zbp{year}detail.txt"
        zbp_totals_file = f"zbp{year}totals.txt"
        cbp_state_file = f"cbp{year}st.txt"
        enow_zips_file = f"enowZips{year}.csv"

        # --- Define explicit column names for raw txt files ---
        zbp_detail_cols = [
            'zip', 'naics', 'est', 'n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 
            'n100_249', 'n250_499', 'n500_999', 'n1000', 'emp', 'qp1', 'ap', 
            'emp_nf', 'qp1_nf', 'ap_nf', 'stabbr', 'cty_name', 'name'
        ]
        zbp_totals_cols = [
            'zip', 'est_totals', 'emp_totals', 'qp1_totals', 'ap_totals', 
            'emp_nf_totals', 'qp1_nf_totals', 'ap_nf_totals', 
            'city', 'stabbr_totals', 'cty_name_totals', 'name_totals'
        ]
        cbp_st_cols = [
            'fipstate', 'naics', 'lfo', 'est', 'emp', 'qp1', 'ap', 'empflag', 
            'emp_nf', 'qp1_nf', 'ap_nf', 'n1_4', 'n5_9', 'n10_19', 'n20_49', 
            'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000'
        ]

        # Load raw data files, applying our own headers
        df_zbp_detail = pd.read_csv(zbp_detail_file, dtype={'zip': str}, header=None, names=zbp_detail_cols, skiprows=1)
        df_zbp_totals = pd.read_csv(zbp_totals_file, dtype={'zip': str}, header=None, names=zbp_totals_cols, skiprows=1)
        df_cbp_state = pd.read_csv(cbp_state_file, header=None, names=cbp_st_cols, skiprows=1)
        df_enow_zips = pd.read_csv(enow_zips_file, dtype={'ZIP': str})
        
        # --- Load reference and setup data from CSVs ---
        df_naics_ref = pd.read_csv("ELMEtemplate.xlsx - REFERENCE_naics.csv")
        df_setup = pd.read_csv("ELMEtemplate.xlsx - SETUP.csv", header=3)
        df_coastal_counties = pd.read_csv("ENOW_Geo_Reference.csv")

        # --- Rename columns programmatically to avoid KeyErrors ---
        if not df_naics_ref.empty:
            df_naics_ref.rename(columns={
                df_naics_ref.columns[0]: 'NAICS Code',
                df_naics_ref.columns[1]: 'NAICS Title',
                df_naics_ref.columns[2]: 'ENOW Sector'
            }, inplace=True)
        
        if not df_setup.empty:
            df_setup.rename(columns={df_setup.columns[2]: 'NAICS Codes'}, inplace=True)

        # Load geospatial data for county maps
        county_geo = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip")
        
        return df_zbp_detail, df_zbp_totals, df_cbp_state, df_enow_zips, df_naics_ref, df_setup, df_coastal_counties, county_geo

    except FileNotFoundError as e:
        st.error(f"Error: Missing data file - {e.filename}. Please make sure all required raw data files for the selected year are in the same directory as the app.")
        return None, None, None, None, None, None, None, None

def perform_employment_estimation(df_zbp_detail, df_zbp_totals, df_cbp_state, df_enow_zips, df_naics_ref, df_coastal_counties):
    """
    Replicates the employment estimation logic from the R Markdown script.
    """
    # --- 1. ZBP Data Cleaning ---
    zbp = df_zbp_detail.copy()
    zbp['zip'] = zbp['zip'].str.zfill(5)
    zbp['naics'] = zbp['naics'].astype(str).str.replace('[-/]', '', regex=True)
    zbp['naics'] = zbp['naics'].replace('------', '0')
    
    # Use the detail file as the main source, rename its 'stabbr' to 'state'
    zbp = zbp.rename(columns={'stabbr': 'state'})
    
    # Use the totals file ONLY to map city names to zip codes
    city_map = df_zbp_totals[['zip', 'city']].drop_duplicates()
    city_map['zip'] = city_map['zip'].str.zfill(5)
    zbp = pd.merge(zbp, city_map, on='zip', how='left')
    
    shoreline_zips_list = df_enow_zips['ZIP'].str.zfill(5).unique()
    zbp['Coastal_Zip'] = np.where(zbp['zip'].isin(shoreline_zips_list), "YES", "NO")
    
    est_cols = ['n1_4', 'n5_9', 'n10_19', 'n20_49', 'n50_99', 'n100_249', 'n250_499', 'n500_999', 'n1000']
    for col in est_cols:
        zbp[col] = pd.to_numeric(zbp[col], errors='coerce')
    
    # --- 2. CBP State Data Cleaning ---
    cbp_state = df_cbp_state[df_cbp_state['lfo'] == '-'].copy()
    cbp_state['naics'] = cbp_state['naics'].astype(str).str.replace('[-/]', '', regex=True)
    cbp_state['naics'] = cbp_state['naics'].replace('------', '0')
    
    for col in est_cols:
        cbp_state[col] = pd.to_numeric(cbp_state[col], errors='coerce')

    # --- 3. Create Establishment Size Distribution ---
    est_dist = cbp_state.groupby('naics')[est_cols].sum(min_count=1)
    row_sums = est_dist.sum(axis=1)
    est_ratios = est_dist.div(row_sums.replace(0, np.nan), axis=0)
    
    # --- 4. Distribute Missing Establishments ---
    zbp['est'] = pd.to_numeric(zbp['est'], errors='coerce')
    zbp['missingEsts'] = zbp['est'] - zbp[est_cols].sum(axis=1)
    
    zbp = pd.merge(zbp, est_ratios.add_suffix('_ratio'), on='naics', how='left')
    
    for col in est_cols:
        ratio_col = f'{col}_ratio'
        zbp[col] = zbp[col].fillna(zbp['missingEsts'] * zbp[ratio_col])
        
    # --- 5. Estimate Employment ---
    size_midpoints = {
        'n1_4': 2.5, 'n5_9': 7.0, 'n10_19': 14.5, 'n20_49': 34.5, 'n50_99': 74.5,
        'n100_249': 174.5, 'n250_499': 374.5, 'n500_999': 749.5, 'n1000': 1500.0
    }
    
    zbp['Estimated Employment'] = 0
    for col, midpoint in size_midpoints.items():
        zbp['Estimated Employment'] += zbp[col].fillna(0) * midpoint
    
    # --- 6. Final Cleanup and Merge ---
    zbp = zbp.rename(columns={'est': 'Establishments'})
    
    df_naics_ref['naics'] = df_naics_ref['NAICS Code'].astype(str)
    zbp = pd.merge(zbp, df_naics_ref[['naics', 'NAICS Title', 'ENOW Sector']], on='naics', how='left')
    zbp['ENOW Sector'] = zbp['ENOW Sector'].fillna('Not Covered')

    zbp = zbp.rename(columns={'cty_name': 'county'})
    if 'state' in zbp.columns:
        # FIX: Convert column to string type before using .str accessor
        zbp['state'] = zbp['state'].astype(str).str.strip()

    final_df = zbp[['zip', 'city', 'state', 'county', 'Coastal_Zip', 'naics', 'NAICS Title', 'ENOW Sector', 'Establishments', 'Estimated Employment']].copy()
    final_df = final_df.dropna(subset=['state', 'county'])

    return final_df.round({'Estimated Employment': 0})


# --- Helper Functions ---
def to_excel(dfs: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    processed_data = output.getvalue()
    return processed_data

def reset_to_step(step_number):
    keys_to_clear = []
    if step_number <= 1: keys_to_clear.extend(['selected_states', 'selected_counties', 'selected_zips', 'selected_naics'])
    if step_number <= 2: keys_to_clear.extend(['selected_counties', 'selected_zips', 'selected_naics'])
    if step_number <= 3: keys_to_clear.extend(['selected_zips', 'selected_naics'])
    if step_number <= 4: keys_to_clear.extend(['selected_naics'])
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]
    st.session_state.step = step_number


# --- Main App ---
def main():
    st.title("🌊 Local Marine Economy Estimator")
    st.markdown("This tool helps you estimate the size and composition of the marine economy in your local area. Follow the steps below to generate your custom analysis.")
    
    year_full = st.selectbox("Select Analysis Year", ["2023", "2022"], key="year_select")
    year_short = year_full[-2:]

    with st.spinner(f"Loading and processing data for {year_full}... This may take a moment."):
        raw_data = load_data(year=year_short)
    
    if any(d is None for d in raw_data):
        st.stop()
    
    df_zbp_detail, df_zbp_totals, df_cbp_state, df_enow_zips, df_naics_ref, df_setup, df_coastal_counties, county_geo = raw_data
    
    df_details = perform_employment_estimation(df_zbp_detail, df_zbp_totals, df_cbp_state, df_enow_zips, df_naics_ref, df_coastal_counties)

    if 'step' not in st.session_state:
        st.session_state.step = 1

    # --- Step 1: State Selection ---
    if st.session_state.step >= 1:
        st.header("Step 1: Select State(s) of Interest")
        
        available_states = sorted(df_details['state'].unique())
        
        if not available_states:
            st.error("No state data could be loaded. Please check that the raw data files (e.g., zbp23detail.txt) are in the correct format and location.")
            st.stop()
        
        if 'last_year' not in st.session_state or st.session_state.last_year != year_full:
            st.session_state.selected_states = []
        st.session_state.last_year = year_full

        selected_states = st.multiselect(
            "Select State(s) of Interest",
            available_states,
            default=st.session_state.get('selected_states', []),
            key="state_multiselect"
        )
        if st.button("Next: Select Counties", type="primary"):
            if selected_states:
                st.session_state.selected_states = selected_states
                st.session_state.step = 2
                st.rerun()
            else:
                st.warning("Please select at least one state.")

    # --- Step 2: County Selection ---
    if st.session_state.step >= 2:
        st.header("Step 2: Select Counties of Interest")
        filtered_by_state = df_details[df_details['state'].isin(st.session_state.selected_states)]
        available_counties = sorted(filtered_by_state['county'].unique())
        selected_counties = st.multiselect(
            "Select Counties",
            available_counties,
            default=st.session_state.get('selected_counties', []),
            key="county_multiselect"
        )

        st.subheader("Map of Selected State(s)")
        with st.spinner("Generating map..."):
            map_data = county_geo[county_geo['STATE_NAME'].isin([s.upper() for s in st.session_state.selected_states])]
            if not map_data.empty:
                map_center = [map_data.unary_union.centroid.y, map_data.unary_union.centroid.x]
                m = folium.Map(location=map_center, zoom_start=6)
                folium.GeoJson(
                    map_data,
                    style_function=lambda feature: {
                        'fillColor': '#228B22' if feature['properties']['NAME'].upper() in [c.upper() for c in selected_counties] else '#D3D3D3',
                        'color': 'black', 'weight': 1,
                        'fillOpacity': 0.6 if feature['properties']['NAME'].upper() in [c.upper() for c in selected_counties] else 0.2,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['NAME'], aliases=['County:'])
                ).add_to(m)
                st_folium(m, use_container_width=True, height=500)
            else:
                st.info("Could not generate map. Geospatial data not found for selected states.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to State Selection"): reset_to_step(1); st.rerun()
        with col2:
            if st.button("Next: Select Zip Codes", type="primary"):
                if selected_counties:
                    st.session_state.selected_counties = selected_counties
                    st.session_state.step = 3
                    st.rerun()
                else:
                    st.warning("Please select at least one county.")

    # --- Step 3: Zip Code Selection ---
    if st.session_state.step >= 3:
        st.header("Step 3: Select Zip Codes of Interest")
        filtered_by_county = df_details[df_details['county'].isin(st.session_state.selected_counties)]
        zip_options = filtered_by_county[['zip', 'city', 'county', 'Coastal_Zip']].drop_duplicates().sort_values(by=['county', 'zip'])
        st.info("The table below shows all available zip codes in your selected counties.")
        selected_zips = st.multiselect(
            "Select Zip Codes",
            options=zip_options['zip'].tolist(),
            default=st.session_state.get('selected_zips', zip_options['zip'].tolist()),
            key="zip_multiselect"
        )
        st.dataframe(zip_options, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to County Selection"): reset_to_step(2); st.rerun()
        with col2:
            if st.button("Next: Customize Industries", type="primary"):
                if selected_zips:
                    st.session_state.selected_zips = selected_zips
                    st.session_state.step = 4
                    st.rerun()
                else:
                    st.warning("Please select at least one zip code.")

    # --- Step 4: NAICS Code Customization ---
    if st.session_state.step >= 4:
        st.header("Step 4: Customize Industries (NAICS Codes)")
        default_naics_list = df_setup['NAICS Codes'].dropna().astype(str).tolist()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your NAICS List")
            naics_input = st.text_area(
                "NAICS Codes for Analysis",
                value="\n".join(st.session_state.get('selected_naics', default_naics_list)),
                height=300
            )
            selected_naics = [code.strip() for code in naics_input.split('\n') if code.strip()]
        with col2:
            st.subheader("Reference: NAICS Sectors")
            st.dataframe(df_naics_ref, height=300, use_container_width=True)
        col_back, col_generate = st.columns(2)
        with col_back:
            if st.button("Back to Zip Code Selection"): reset_to_step(3); st.rerun()
        with col_generate:
            if st.button("Generate Analysis", type="primary"):
                if selected_naics:
                    st.session_state.selected_naics = selected_naics
                    st.session_state.step = 5
                    st.rerun()
                else:
                    st.warning("Please provide at least one NAICS code.")

    # --- Step 5: Results ---
    if st.session_state.step >= 5:
        st.header("📈 Analysis Results")
        final_selection = df_details[
            (df_details['zip'].isin(st.session_state.selected_zips)) &
            (df_details['naics'].isin(st.session_state.selected_naics))
        ].copy()
        
        if final_selection.empty:
            st.warning("No data found for the selected criteria. Please go back and adjust your selections.")
        else:
            metrics = ['Establishments', 'Estimated Employment']
            table1 = final_selection.groupby(['zip', 'city', 'county'])[metrics].sum().reset_index()
            table2 = final_selection.groupby('county')[metrics].sum().reset_index()
            table3 = final_selection.groupby(['naics', 'NAICS Title', 'ENOW Sector'])[metrics].sum().reset_index()
            table4_est = pd.pivot_table(final_selection, values='Establishments', index='county', columns='ENOW Sector', aggfunc='sum', fill_value=0)
            table4_emp = pd.pivot_table(final_selection, values='Estimated Employment', index='county', columns='ENOW Sector', aggfunc='sum', fill_value=0)
            table5_est = pd.pivot_table(final_selection, values='Establishments', index=['zip', 'city', 'county'], columns='ENOW Sector', aggfunc='sum', fill_value=0)
            table5_emp = pd.pivot_table(final_selection, values='Estimated Employment', index=['zip', 'city', 'county'], columns='ENOW Sector', aggfunc='sum', fill_value=0)

            st.subheader("Table 1: Marine Economy Summary by Zip Code")
            st.dataframe(table1, use_container_width=True)
            st.subheader("Table 2: Marine Economy Summary by County")
            st.dataframe(table2, use_container_width=True)
            st.subheader("Table 3: Marine Economy Summary by Industry (NAICS)")
            st.dataframe(table3, use_container_width=True)
            st.subheader("Table 4: Detailed Industry Composition by County")
            st.markdown("Establishments:"); st.dataframe(table4_est, use_container_width=True)
            st.markdown("Estimated Employment:"); st.dataframe(table4_emp, use_container_width=True)
            st.subheader("Table 5: Detailed Industry Composition by Zip Code")
            st.markdown("Establishments:"); st.dataframe(table5_est, use_container_width=True)
            st.markdown("Estimated Employment:"); st.dataframe(table5_emp, use_container_width=True)

            excel_data = to_excel({
                "Table1_by_Zip": table1, "Table2_by_County": table2, "Table3_by_NAICS": table3,
                "Table4_County_Est": table4_est, "Table4_County_Emp": table4_emp,
                "Table5_Zip_Est": table5_est, "Table5_Zip_Emp": table5_emp
            })
            st.download_button(
                label="📥 Download All Tables as Excel", data=excel_data,
                file_name="marine_economy_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if st.button("Start Over"): reset_to_step(1); st.rerun()

if __name__ == "__main__":
    main()
