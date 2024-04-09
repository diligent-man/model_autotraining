import re


def replace_json_placeholders(json, values):
    # find all placeholders
    placeholders = re.findall('<[\w ]+>', json)
    # clear_placeholders = list(map(lambda x: x.replace('<', '').replace('>', ''), placeholders))

    assert len(placeholders) == len(values), "Please enter the values of all placeholders."

    # replaces all placeholders with values
    for k, v in values.items():
        placeholder = "<%s>" % k
        json = json.replace(placeholder, v)

    return json


# Example
json = "{'firstName':'<first_name>','lastName':'<last_name>','country':'Turkey','city':'Istanbul'}"
values = {'first_name': 'muhammet', 'last_name': 'guner'}
print(replace_json_placeholders(json, values))