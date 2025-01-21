import json

def read_json_file(file_path):
    """
    Reads a JSON file and returns its content.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_json_file(data, file_path):
    """
    Saves a dictionary to a JSON file.

    Parameters:
        data (dict): The data to be saved.
        file_path (str): The path to save the JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
            print(f"Data successfully saved to '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")

# Example usage
json_content_P = read_json_file(r"E:\PADS\Projects\IMr_pos_neg\outputs\test.json")
# cp = {(x["template"],x["parameters"]) for x in json_content_P["constraints"]}
c = set()
for x in json_content_P["constraints"]:
    if len(x["parameters"])>1:
        c.add((x["template"], x["parameters"][0][0],x["parameters"][1][0]))
    else:
        c.add((x["template"], x["parameters"][0][0]))
json_content_M = read_json_file(r"E:\PADS\Projects\IMr_pos_neg\outputs\test.json")
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



print(json_content)

# not_included = {
#                  'NotRespondedExistence',
#                  'NotPrecedence',
#                  'NotChainPrecedence',
#                  'NotResponse',
#                  'NotChainResponse',
#                  'AtLeast2',
#                  'AtLeast3',
#                  'AtMost2',
#                  'AtMost3',
#                  }
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

const_filtered = []
for c in json_content['constraints']:
    if c['template'] in mapping.keys():
        new_c = c
        new_c['template'] = mapping[c['template']]
        const_filtered.append(new_c)


json_content['constraints'] = const_filtered

save_json_file(json_content, r"E:\PADS\Projects\IMr_pos_neg\outputs\test_filtered.json")
print("hey")
