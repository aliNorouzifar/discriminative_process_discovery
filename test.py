from functions.utils import read_json_file
import pm4py
# from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
import pandas as pd

def extract_trace_variants(xes_file_path):
    """
    Reads an XES event log file and extracts trace variants using pm4py.

    Parameters:
        xes_file_path (str): Path to the XES file.

    Returns:
        dict: A dictionary where keys are trace variants and values are their frequencies.
    """
    # Import the XES event log
    log = pm4py.read_xes(xes_file_path)

    # Extract trace variants and their frequencies
    variants = variants_filter.get_variants(log)

    # Count frequencies of each variant
    variant_counts = {"\u003c" + ",".join(variant) +"\u003e": len(instances) for variant, instances in variants.items()}

    return variant_counts

# Example usage
xes_file_path = "LP.xes"
trace_variants = extract_trace_variants("E:\PADS\Projects\IMr_pos_neg\outputs\LP.xes")
print(trace_variants)

data = read_json_file(r"E:\PADS\Projects\IMr_pos_neg\outputs\xx[eventsEvaluation].json")

for tr in data:
    for constraint in data[tr]:
        if 2 in data[tr][constraint]:
            data[tr][constraint] = "violated"
        elif 3 in data[tr][constraint]:
            data[tr][constraint] = "satisfied"
        else:
            data[tr][constraint] = "vac_satisfied"

df = pd.DataFrame(data).T
df_count =pd.DataFrame(trace_variants)
repeated_indices = df.index.repeat(df_count)
result = df.loc[repeated_indices].reset_index()
print('s')