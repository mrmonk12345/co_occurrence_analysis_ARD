

import pandas as pd # Import pandas for data manipulation



def clean_taxa_df(df):
    """
    Clean string columns in a taxonomy DataFrame.

    Replaces spaces, pipes (`|`), and hyphens (`-`) with dots (`.`) in all string columns.
    The operation is applied in place and returns the modified DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing taxonomy data.

    Returns:
        pd.DataFrame: The cleaned DataFrame with updated string columns.
    """
    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].apply(
        lambda col: col.str.replace(r"[ \|\-]", ".", regex=True)
    )
    return df


# Cleans up and merges data for relative abundance bar plots
def clean_and_merge_data(otu_file, taxon_file, drop_past=True):
    """
    cleans and merges OTU and taxonomy data files.
    Args:
        otu_file (str): Path to the OTU CSV file.
        taxon_file (str): Path to the Taxonomy CSV file.
        drop_past (bool): Whether to drop columns starting with 'Past'. Defaults to True.
    
    Returns:
        pd.DataFrame: Merged and cleaned DataFrame.
    """
    # Load the datasets
    otu_df = pd.read_csv(otu_file, index_col=0)

    taxon_df = pd.read_csv(taxon_file, index_col=0)


    # Merge OTU and taxonomy tables on feature ID
    merged_df = otu_df.merge(taxon_df, left_index=True, right_index=True)

    # Drop columns starting with 'Past' from merged_df

    if(drop_past == True):
        cols_to_drop_merged = [c for c in merged_df.columns if c.lower().startswith("past")]
        merged_df = merged_df.drop(columns=cols_to_drop_merged)

    # Create new column for full taxonomy
    taxonomy_levels = ["Kingdom", "Phylum", "Class", "Order", "Family","Genus", "Species"]
    merged_df["Full_Taxonomy"] = merged_df[taxonomy_levels].agg(";".join, axis=1)

    # Merge items with same full taxonomy
    grouped_df = merged_df.copy()
    grouped_df = grouped_df.groupby("Full_Taxonomy").sum()
    grouped_df = grouped_df.reset_index()

    grouped_df = grouped_df.drop(columns=taxonomy_levels)

    # Sort by taxonomy levels
    tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    grouped_df[tax_levels] = grouped_df["Full_Taxonomy"].str.extract(
        r"(k__[^;]*);(p__[^;]*);(c__[^;]*);(o__[^;]*);(f__[^;]*);(g__[^;]*);(s__.*)",
        expand=True
    )

    grouped_df_sorted = grouped_df.sort_values(by=tax_levels)

    grouped_df_sorted = grouped_df_sorted.reset_index(drop=True)

    return grouped_df_sorted

# Clean and merge for the old data
def clean_and_merge_old_data(otu_file, taxon_file):
    """
    Cleans and merges OTU and taxonomy data files for old data.
    Args:
        otu_file (str): Path to the OTU CSV file.
        taxon_file (str): Path to the Taxonomy CSV file.
        (always drops 'Past' columns for old data)
    Returns:
        pd.DataFrame: Merged and cleaned DataFrame.
    """
    # Load the datasets
    otu_df = pd.read_csv(otu_file, index_col=0)

    taxon_df = pd.read_csv(taxon_file, index_col=0)

    cols_to_keep_otu = [col for col in otu_df.columns if col.startswith('BjSa.G210') or col.startswith('NTC.G210') or col == 'otu name']
    otu_df = otu_df[cols_to_keep_otu]

    taxon_df = taxon_df.drop('Confidence', axis=1)

    # Merge OTU and taxonomy tables on feature ID
    merged_df = otu_df.merge(taxon_df, left_index=True, right_index=True)

    # Drop columns starting with 'Past' from merged_df
    cols_to_drop_merged = [c for c in merged_df.columns if c.lower().startswith("past")]
    merged_df = merged_df.drop(columns=cols_to_drop_merged)

    # Remove spaces after semicolons in the 'Taxon' column
    merged_df['Taxon'] = merged_df['Taxon'].str.replace('; ', ';')


    # Merge items with same full taxonomy
    grouped_df = merged_df.copy()
    grouped_df = grouped_df.groupby("Taxon").sum()
    grouped_df = grouped_df.reset_index()


    grouped_df = grouped_df[grouped_df.sum(axis=1, numeric_only=True) > 0]

    tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    grouped_df[tax_levels] = grouped_df["Taxon"].str.extract(
        r"d__([^;]*);p__([^;]*);c__([^;]*);o__([^;]*);f__([^;]*);g__([^;]*);s__(.*)"
    )

    grouped_df_sorted = grouped_df.sort_values(by=tax_levels)

    grouped_df_sorted = grouped_df_sorted.reset_index(drop=True)

    # Drop 'otu name' and 'Feature.ID' columns
    columns_to_drop = ['otu name', 'Feature.ID']
    existing_columns = [col for col in columns_to_drop if col in grouped_df_sorted.columns]
    grouped_df_sorted = grouped_df_sorted.drop(columns=existing_columns)

    grouped_df_sorted.rename(columns={"Taxon": "Full_Taxonomy"}, inplace=True)
    return grouped_df_sorted

# Function to compute relative abundance
def compute_relative_abundance(df, sample_cols, per='sample', scale=1):
    """
    Convert sample columns to relative abundance.

    Normalizes values either per sample or per feature and optionally scales them.

    Args:
        data (pd.DataFrame): Input data with samples as columns and features as rows.
        per (str): Normalization mode.
            - 'sample': Divide each column by its column sum (per-sample normalization).
            - 'row': Divide each row by its row sum (per-feature normalization).
        scale (float): Factor to multiply normalized values by (e.g., 100 for percentages).

    Returns:
        pd.DataFrame: A copy of the input data with normalized values.
    """
    rdf = df.copy()
    rdf[sample_cols] = rdf[sample_cols].astype(float)  # Ensure float dtype
    
    if per == 'sample':
        denom = rdf[sample_cols].sum(axis=0)
        rdf.loc[:, sample_cols] = rdf[sample_cols].div(denom, axis=1).fillna(0) * scale
    else:
        denom = rdf[sample_cols].sum(axis=1)
        rdf.loc[:, sample_cols] = rdf[sample_cols].div(denom, axis=0).fillna(0) * scale
    return rdf


def add_mean_relative_abundance(df, sample_cols = None):
    # Get sample columns (those with "-")
    if sample_cols is None:
        sample_cols = [c for c in df.columns if "-" in c]

    # Add average
    # Recalculate total for final_df after concatenation
    df['total'] = df[sample_cols].sum(axis=1)
    df['mean_relative_abundance'] = df['total'] / len(sample_cols) # Use actual number of sample columns
    return df


# function to make phylum_df , class_df, order_df , family_df, genus_df , species_df

def make_taxon_df(df, taxon_level, relative=False, per='sample', scale=1):
    """
    combines rows by taxonomic level and sum abundance, optionally converts to relative abundance.
    Args:
        df (pd.DataFrame): Input DataFrame with taxonomy and abundance data.
        taxon_level (str): Taxonomic level to group by (e.g., 'Phylum', 'Class', etc.).
        relative (bool): Whether to convert to relative abundance. Defaults to False.
        per (str): Normalization mode for relative abundance ('sample' or 'row'). Defaults to 'sample'.
        scale (float): Scaling factor for relative abundance. Defaults to 1.
    
    Returns:
        pd.DataFrame: DataFrame grouped by taxon_level with summed abundance and optional relative abundance.
    """
    tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]




    # sample columns = columns that are not taxonomy or Full_Taxonomy
    sample_cols = [c for c in df.columns if c not in tax_levels + ["Full_Taxonomy"]]

    # ensure level exists (if you extracted taxonomy earlier it should)
    if taxon_level not in df.columns:
        raise KeyError(f"{taxon_level} column not found in grouped_df_sorted")


     # aggregation: sum sample cols, keep taxonomy columns by first()
    agg = {c: "sum" for c in sample_cols}
    for t in tax_levels:
        if t in df.columns and t != taxon_level:
            agg[t] = "first"


    # group by the requested level; sort=False preserves first-appearance order
    grouped = df.groupby(taxon_level, sort=False).agg(agg)

    # reset index so taxon_level becomes a column
    result = grouped.reset_index()

    # reorder columns: level, sample cols, higher taxonomy (keep_levels), extras
    current_idx = tax_levels.index(taxon_level)
    keep_levels = tax_levels[:current_idx]
    cols = [taxon_level] + sample_cols + keep_levels
    cols = [c for c in cols if c in result.columns]
    result = result.loc[:, cols]

    # optionally convert to relative abundance (per-sample default)
    if relative:
        result = compute_relative_abundance(result, sample_cols, per=per, scale=scale)


    return result

# function to make the final taxon dataframe
def make_final_taxon_df(taxon_df, level, top_n=10, order="taxonomy", combine_other_unknown=True, sample_cols = None):
    """
    Create a DataFrame summarizing taxa abundance.

    The resulting DataFrame includes:
        - Top N taxa by abundance.
        - An 'Other' row for remaining taxa.
        - An 'Unknown' row if blank or missing values exist.
    Preserves taxonomy columns and original order. Optionally, 'Other' and 'Unknown' rows
    can remain separate instead of being combined.

    Args:
        data (pd.DataFrame): Input data with taxa information.
        top_n (int): Number of top taxa to include.
        combine_other (bool): Whether to combine 'Other' and 'Unknown' rows into one.

    Returns:
        pd.DataFrame: A summarized DataFrame with top taxa and aggregated rows.
    """
    # Get sample columns (those with "-")
    if sample_cols is None:
        sample_cols = [c for c in taxon_df.columns if "-" in c]

    # Work with a copy
    df = taxon_df.copy()

    # Handle unknown/blank values in level column
    df[level] = df[level].fillna("").astype(str).str.strip()
    unknown_mask = df[level] == ""

    # Split into known and unknown
    known_df = df[~unknown_mask].copy()
    unknown_df = df[unknown_mask].copy()

    # Get top N by total abundance
    known_df['total'] = known_df[sample_cols].sum(axis=1)
    # Get top N indices while preserving order
    sorted_idx = known_df['total'].nlargest(top_n).index
    top_n_df = known_df.loc[sorted_idx].sort_index()  # Sort by original index to preserve order

    other_df = known_df[~known_df.index.isin(sorted_idx)]

    # Sum values for Other row
    other_sums = other_df[sample_cols].sum()

    # Create Other row (all taxonomy columns blank)
    other_row = pd.DataFrame([{
        level: 'Other',
        **{col: "" for col in df.columns if col not in sample_cols and col != level},
        **{col: other_sums[col] for col in sample_cols}
    }])

    # Sum values for Unknown row
    unknown_row = pd.DataFrame() # Initialize unknown_row as empty
    if not unknown_df.empty:
        unknown_sums = unknown_df[sample_cols].sum()
        # Create Unknown row (all taxonomy columns blank)
        unknown_row = pd.DataFrame([{
            level: 'Unknown',
            **{col: "" for col in df.columns if col not in sample_cols and col != level},
            **{col: unknown_sums[col] for col in sample_cols}
            }])

    # Drop the temporary total column from top_n_df
    if 'total' in top_n_df.columns:
        top_n_df = top_n_df.drop(columns=['total'])


    # Combine all parts
    if combine_other_unknown:
        # Combine sums from both rows if unknown_df is not empty
        if not unknown_df.empty:
            combined_sums = other_sums + unknown_sums
            other_row = pd.DataFrame([{
                level: 'Other', # Label as 'Other' when combined
                **{col: "" for col in df.columns if col not in sample_cols and col != level},
                **{col: combined_sums[col] for col in sample_cols}
            }])
        final_df = pd.concat([top_n_df, other_row], ignore_index=True)
    else:
        # Concatenate top_n_df, other_row, and unknown_row separately
        final_df = pd.concat([top_n_df, other_row, unknown_row], ignore_index=True)


    # Add average
    # Recalculate total for final_df after concatenation
    final_df['total'] = final_df[sample_cols].sum(axis=1)
    final_df['mean_relative_abundance'] = final_df['total'] / len(sample_cols) # Use actual number of sample columns


    if (order=='abundance'):
        final_df.sort_values(by='mean_relative_abundance', ascending=False, inplace=True)


    # Move "Other" and "Unknown" rows to the bottom if not combined
    if not combine_other_unknown:
        rows_to_move = final_df[final_df[level].isin(["Other", "Unknown"])].copy()
        final_df = final_df[~final_df[level].isin(["Other", "Unknown"])].copy()
        final_df = pd.concat([final_df, rows_to_move], ignore_index=True)
    else:
         # Move only "Other" to the bottom if combined
         row_to_move = final_df.loc[final_df[level] == "Other"].copy()
         final_df = final_df.loc[final_df[level] != "Other"].copy()
         final_df = pd.concat([final_df, row_to_move], ignore_index=True)



    return final_df


# Final Ordering

# Append the 'Other' row to add it back later
def append_other_rows(df):
    """
    Separates 'Other' rows from the first column.

    Returns: (df_without_other, other_rows)
    """
    first_col = df.columns[0]
    other_rows = df[df[first_col] == 'Other'].copy()
    df_without_other = df[df[first_col] != 'Other'].copy()

    return df_without_other, other_rows

# Append the 'Unknown' row to add it back later
def append_unknown_rows(df):
    """
    Separates 'Unknown' rows from the first column.

    Returns: (df_without_unknown, unknown_rows)
    """
    first_col = df.columns[0]
    unknown_rows = df[df[first_col] == 'Unknown'].copy()
    df_without_unknown = df[df[first_col] != 'Unknown'].copy()

    return df_without_unknown, unknown_rows

# Append both 'Unknown' and 'Other' rows to add them back later
def append_unknown_and_other_rows(df):
    """
    Separates 'Other' and 'Unknown' rows from the first column.

    Returns: (df_clean, other_rows, unknown_rows)
    """

    df_clean, other_rows = append_other_rows(df)
    df_clean, unknown_rows = append_unknown_rows(df_clean)

    return df_clean, other_rows, unknown_rows


# Function to filter and sort primary-level data based on a specified taxonomy level
def filter_and_sort_by_taxonomy(primary_df, taxonomy_df, taxonomy_level: str):
    """
    Filters and sorts primary-level data based on the abundance of a specified taxonomy level.
    Keeps unmatched taxonomy entries (present in primary_df but not in taxonomy_df)
    and appends them at the bottom.

    Args:
        primary_df (pd.DataFrame): Primary-level DataFrame containing taxonomy and abundance data.
        taxonomy_df (pd.DataFrame): DataFrame containing taxonomy and mean relative abundance data.
        taxonomy_level (str): The taxonomy level to filter and sort by.
    
    Returns:
        pd.DataFrame: Filtered and sorted DataFrame.
    """

    # Get the first column name from the primary file
    identifier_col = primary_df.columns[0]

    # Split into matching and non-matching taxa
    matching_df = primary_df[primary_df[taxonomy_level].isin(taxonomy_df[taxonomy_level])]
    missing_df = primary_df[~primary_df[taxonomy_level].isin(taxonomy_df[taxonomy_level])].copy()

    # Merge to get mean abundance (only for matching taxonomy)
    merged_df = matching_df.merge(
        taxonomy_df[[taxonomy_level, 'mean_relative_abundance']],
        on=taxonomy_level,
        how='left'
    )

    # Identify the correct abundance column
    abundance_cols = [col for col in merged_df.columns if 'mean_relative_abundance' in col and col != 'mean_relative_abundance_x']
    if abundance_cols:
        abundance_col = abundance_cols[0]
    else:
        raise KeyError("Could not find mean_relative_abundance column after merge.")

    # Rename for clarity
    merged_df = merged_df.rename(columns={abundance_col: f'{taxonomy_level}_mean_abundance'})

    # Sort by abundance
    sorted_df = merged_df.sort_values(by=f'{taxonomy_level}_mean_abundance', ascending=False)

    # Rename mean_relative_abundance_x → mean_relative_abundance
    if 'mean_relative_abundance_x' in sorted_df.columns:
        sorted_df = sorted_df.rename(columns={'mean_relative_abundance_x': 'mean_relative_abundance'})

    # Append missing taxa (those not in taxonomy_df) at the bottom
    if not missing_df.empty:
        # Add placeholder abundance for missing taxa
        missing_df[f'{taxonomy_level}_mean_abundance'] = pd.NA
        # Match columns for consistent concatenation
        for col in sorted_df.columns:
            if col not in missing_df.columns:
                missing_df[col] = pd.NA
        # Append at bottom
        sorted_df = pd.concat([sorted_df, missing_df[sorted_df.columns]], ignore_index=True)

    return sorted_df


# recursively splits the dataframe by taxonomy levels, reads csvs and needs a dictionary for csvs 
"""
Dictionary Example: df_dict = 
{'Phylum' : 'Phylum_top10_Old_Data.csv', 
'Class' : 'Class_top15_Old_Data.csv', 
'Order' : 'Order_top15_Old_Data.csv', 
'Family' : 'Family_top15_Old_Data.csv', 
'Genus' : 'Genus_top15_Old_Data.csv', 
'Species' : 'Species_top15_Old_Data.csv'}

"""
def recursive_split(df, levels, df_dict):
    """
    Recursively split a dataframe by a list of column names.
    Returns a nested dictionary of dataframes.

    Args:
        df (pd.DataFrame): The input dataframe to split.
        levels (list): List of column names to split by.
        df_dict (dict): Dictionary mapping taxonomy levels to CSV file paths.
    
    Returns:
        dict or pd.DataFrame: Nested dictionary of dataframes or a dataframe if no levels left
    """
    if not levels:
        df = df.sort_values(by='mean_relative_abundance', ascending=False)
        return df



    level = levels[0]


    df = filter_and_sort_by_taxonomy(df, pd.read_csv(df_dict[level]), level)

    grouped = df.groupby(level, sort = False)
    return {key: recursive_split(group, levels[1:], df_dict) for key, group in grouped}

# concatenate nested dictionary of dataframes into a single dataframe
def flatten_nested_dict(nested):
    result = []
    def collect_leaves(d):
        for value in d.values():
            if isinstance(value, dict):
                collect_leaves(value)  # Go deeper
            else:
                result.append(value)  # Collect DataFrame
    collect_leaves(nested)
    return pd.concat(result, ignore_index=True)

# function to order a csv
def final_order_csv(file_path, df_dict, level):
    """
    Processes a CSV file by cleaning, grouping, and merging data based on taxonomic levels.

    Args:
        file_path (str): Path to the input CSV file.
        df_dict (dict): Dictionary used for recursive grouping.
        level (str): Taxonomic level to split by (e.g., 'Order').

    Returns:
        pd.DataFrame: Final merged DataFrame.
    """

    # Define the full hierarchy of taxonomic levels
    taxonomic_levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # Validate the level
    if level not in taxonomic_levels:
        raise ValueError(f"Invalid level '{level}'. Must be one of: {taxonomic_levels}")

    # Determine which levels to use based on the chosen level
    selected_levels = taxonomic_levels[:taxonomic_levels.index(level)]

    # Read the CSV file
    taxon_df = pd.read_csv(file_path)

    # Apply mode-specific filtering or transformation
    taxon_clean, other_rows, unknown_rows = append_unknown_and_other_rows(taxon_df)

    # Recursively group the cleaned DataFrame
    nested_groups = recursive_split(taxon_clean, selected_levels, df_dict)

    # Flatten the nested dictionary into a DataFrame
    merged_df = flatten_nested_dict(nested_groups)

    # Append 'Other' rows back to the final DataFrame
    final_df = pd.concat([merged_df, other_rows], ignore_index=True)

    return final_df



