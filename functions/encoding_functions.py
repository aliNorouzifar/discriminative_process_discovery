from pm4py.algo.filtering.log.variants import variants_filter
import pandas as pd
import pm4py


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


def encode(trace_variants,data):
    for tr in data:
        for constraint in data[tr]:
            if constraint != "MODEL":
                if 2 in data[tr][constraint]:
                    data[tr][constraint] = "violated"
                elif 3 in data[tr][constraint]:
                    data[tr][constraint] = "satisfied"
                else:
                    data[tr][constraint] = "vac_satisfied"

    df = pd.DataFrame(data).T
    df_count =pd.Series(trace_variants)
    repeated_indices = df.index.repeat(df_count)
    result = df.loc[repeated_indices].reset_index()
    return result









