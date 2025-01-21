import csv

import pandas as pd

from functions.subprocess_calls import discover_declare,measurement_extraction
from functions.utils import read_json_file, save_json_file, import_csv_as_dataframe
from functions.encoding_functions import combine_models,create_unique_trace_constraint_df, create_pivot_dataframe
from functions.classification import train_and_evaluate_with_shap


LP_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\LP.xes"
decl_P_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_P.json"
meas_P_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_P.csv"
encoded_P_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\encoded_P.csv"

LM_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\LM.xes"
decl_M_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_M.json"
meas_M_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_M.csv"
encoded_M_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\encoded_M.csv"

combined_model_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_filtered.json"
#
#
# discover_declare(LP_path,decl_P_path)
# discover_declare(LM_path,decl_M_path)
#
# decl_P = read_json_file(decl_P_path)
# decl_M = read_json_file(decl_M_path)
#
# json_content = combine_models(decl_P,decl_M)
#
# save_json_file(json_content, combined_model_path)
#
# measurement_extraction(LP_path,combined_model_path,meas_P_path)
# measurement_extraction(LM_path,combined_model_path,meas_M_path)
#
# # Example usage
# df_P = import_csv_as_dataframe(meas_P_path.removesuffix(".csv")+"[eventsEvaluation].csv")
# result_df_P = create_unique_trace_constraint_df(df_P)
# pivot_df_P = create_pivot_dataframe(result_df_P)
# pivot_df_P.to_csv(encoded_P_path, index=False, sep=';')
#
# df_M = import_csv_as_dataframe(meas_M_path.removesuffix(".csv")+"[eventsEvaluation].csv")
# result_df_M = create_unique_trace_constraint_df(df_M)
# pivot_df_M = create_pivot_dataframe(result_df_M)
# pivot_df_M.to_csv(encoded_M_path, index=False, sep=';')




pivot_df_P = pd.read_csv(encoded_P_path, sep=';')
pivot_df_P = pivot_df_P.loc[pivot_df_P.index.repeat(pivot_df_P['count'])].reset_index(drop=True)
pivot_df_P = pivot_df_P.drop(columns=['count'])


pivot_df_M = pd.read_csv(encoded_M_path, sep=';')
pivot_df_M = pivot_df_M.loc[pivot_df_M.index.repeat(pivot_df_M['count'])].reset_index(drop=True)
pivot_df_M = pivot_df_M.drop(columns=['count'])

train_and_evaluate_with_shap(pivot_df_P, pivot_df_M)
print("stop")
