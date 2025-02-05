from functions.subprocess_calls import discover_declare,measurement_extraction
from functions.utils import read_json_file, save_json_file, import_csv_as_dataframe
from functions.encoding_functions import combine_models,extract_trace_variants,encode
from functions.classification import train_and_evaluate_with_shap


LP_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\LP.xes"
decl_P_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_P.json"
meas_P_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_P_mes.json"
encoded_P_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\encoded_P.csv"

LM_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\LM.xes"
decl_M_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_M.json"
meas_M_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_M_mes.json"
encoded_M_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\encoded_M.csv"

combined_model_path = r"E:\PADS\Projects\IMr_pos_neg\outputs\test_filtered.json"


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


trace_variants_P = extract_trace_variants(LP_path)
data_P = read_json_file(meas_P_path.removesuffix(".json")+"[eventsEvaluation].json")
df_P = encode(trace_variants_P, data_P)


trace_variants_M = extract_trace_variants(LM_path)
data_M = read_json_file(meas_M_path.removesuffix(".json")+"[eventsEvaluation].json")
df_M = encode(trace_variants_M, data_M)


train_and_evaluate_with_shap(df_P, df_M)
print("stop")
