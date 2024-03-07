import re
from functools import reduce
from json import JSONDecoder, JSONDecodeError

NOT_WHITESPACE = re.compile(r'\S')


# Support funcs
def json_decoder(document: str, pos=0, decoder=JSONDecoder()):
    """
    Acceptable format for document:
        a/ Format 1: Single json obj
            '''{obj}'''
        b/ Format 2: Multiple json objs
            '''{obj_1},
               {obj_2},
               {obj_3}
            '''
        c/ Format 2: List of Single or Multiple json objs
            '''
            [{obj_1},
               {obj_2},
               {obj_3}]
            '''
    """
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError as e:
            print(e)
            # do something sensible if there's some error
            raise
        yield obj


def merge_nested_dict(default_config: dict, custom_config: dict, path: list = []):
    # Merge two nested dict together with recursion
    # Everything in dict b will be merged into dict a & overwrite dict_a[key] if dict_a[key] = dict_b[key]
    for key in custom_config:
        if key in default_config:
            if isinstance(default_config[key], dict) and isinstance(custom_config[key], dict):
                merge_nested_dict(default_config[key], custom_config[key], path + [str(key)])
            elif default_config[key] != custom_config[key]:
                default_config[key] = custom_config[key]
                # raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            default_config[key] = custom_config[key]
    return default_config


# Use functions
def get_config(default_config_path: str, custom_config_path: str):
    with open(file=default_config_path, mode="r", encoding="UTF-8", errors="ignore") as f:
        default_config = next(iter(json_decoder(f.read())))

    with open(file=custom_config_path, mode="r", encoding="UTF-8", errors="ignore") as f:
        custom_config = next(iter(json_decoder(f.read())))
    return reduce(merge_nested_dict, [default_config, custom_config])