
import pandas as pd
import ast
from functions.utils import import_csv_as_dataframe

mapping = {
           'Absence':"Absence",
           'AtLeast1':"Participation",
           'AtMost1':"AtMostOne",
           'Init':"Init",
           'End':"End",
           'RespondedExistence':"RespondedExistence",
           'Response':"Response",
           'AlternateResponse':"AlternateResponse",
           'ChainResponse':"ChainResponse",
           'Precedence':"Precedence",
           'AlternatePrecedence':"AlternatePrecedence",
           'ChainPrecedence':"ChainPrecedence",
           "CoExistence":"CoExistence",
           'Succession':"Succession",
           'AlternateSuccession':"AlternateSuccession",
           'ChainSuccession':"ChainSuccession",
           'NotSuccession':"NotSuccession",
           'NotCoExistence':"NotCoExistence",
           'NotChainSuccession':"NotChainSuccession"}

def combine_models(json_content_P,json_content_M):

# cp = {(x["template"],x["parameters"]) for x in json_content_P["constraints"]}
    c = set()
    for x in json_content_P["constraints"]:
        if len(x["parameters"])>1:
            c.add((x["template"], x["parameters"][0][0],x["parameters"][1][0]))
        else:
            c.add((x["template"], x["parameters"][0][0]))
    for x in json_content_M["constraints"]:
        if len(x["parameters"])>1:
            c.add((x["template"], x["parameters"][0][0],x["parameters"][1][0]))
        else:
            c.add((x["template"], x["parameters"][0][0]))
    json_content = {}
    json_content["name"] = "merged_PN"
    json_content["tasks"] = list(set(json_content_P["tasks"]).union(set(json_content_M["tasks"])))
    json_content["constraints"] = []
    for const in c:
        dict = {"template":const[0], "parameters":[[x] for x in const[1:]]}
        json_content["constraints"].append(dict)


    const_filtered = []
    for c in json_content['constraints']:
        if c['template'] in mapping.keys():
            new_c = c
            new_c['template'] = mapping[c['template']]
            const_filtered.append(new_c)

    json_content['constraints'] = const_filtered

    return json_content





def create_unique_trace_constraint_df(df):
    """
    Creates a new DataFrame with unique trace and declarative constraint combinations,
    retains all extra columns from the original DataFrame, and counts their occurrences.

    Additionally, processes the "Events-Evaluation" column (a stringified list) to create separate columns
    for the counts of 0s, 1s, 2s, and 3s. Rows where the second column has the value "MODEL" are excluded.

    Parameters:
        df (pd.DataFrame): The original DataFrame with at least two columns: trace and declarative constraint.
        output_file_path (str): The path to save the resulting DataFrame as a CSV file.

    Returns:
        pd.DataFrame: A new DataFrame with unique combinations, all extra columns, and their counts.
    """
    if df.shape[1] < 2:
        print("Error: The DataFrame must have at least two columns: trace and declarative constraint.")
        return None

    # Exclude rows where the second column has the value "MODEL"
    df = df[df.iloc[:, 1] != "MODEL"].copy()

    group_cols = df.columns[:2].tolist()
    result_df = (
        df.groupby(group_cols, as_index=False)
        .apply(lambda x: x.iloc[0])  # Retain the first row of each group
        .assign(count=df.groupby(group_cols).size().values)  # Add the count column
        .reset_index(drop=True)
    )

    if 'Events-Evaluation' in result_df.columns:
        # Parse the stringified list and count occurrences of 0s, 1s, 2s, and 3s
        result_df[['count_0', 'count_1', 'count_2', 'count_3']] = (
            result_df['Events-Evaluation']
            .apply(lambda x: pd.Series([
                ast.literal_eval(x).count(0),
                ast.literal_eval(x).count(1),
                ast.literal_eval(x).count(2),
                ast.literal_eval(x).count(3)
            ]))
        )
        # Drop the original "Events-Evaluation" column
        result_df = result_df.drop(columns=['Events-Evaluation'])

    # Save the resulting DataFrame as a CSV file
    result_df

    return result_df
def create_pivot_dataframe(result_df):
    """
    Creates a pivot DataFrame where rows are the values from the first column,
    columns are the values from the second column, and each cell is a tuple (x, y):
        - x: The value of the count column
        - y: "violated" if count_2 > 0, "satisfied" if count_3 > 0, otherwise "vacsatisfied".

    Parameters:
        result_df (pd.DataFrame): The processed DataFrame.

    Returns:
        pd.DataFrame: A pivoted DataFrame with the first column as the row index.
    """
    def determine_status(row):
        if row['count_2'] > 0:
            return "violated"
        elif row['count_3'] > 0:
            return "satisfied"
        else:
            return "vac_satisfied"

    result_df['status'] = result_df.apply(determine_status, axis=1)
    result_df['cell_value'] = result_df.apply(lambda row: row['status'], axis=1)

    pivot_df = result_df.pivot(index=result_df.columns[0], columns=result_df.columns[1], values='cell_value').reset_index()
    pivot_df = pivot_df.merge(result_df[['Trace', 'count']].drop_duplicates(), on='Trace', how='left')
    return pivot_df


# Example usage
# df = import_csv_as_dataframe('example.csv')
# result_df = create_unique_trace_constraint_df(df, 'result.csv')








