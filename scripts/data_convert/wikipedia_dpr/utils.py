import json

'''
A simple streaming json reader
'''
def dpr_json_reader(file_to_read):
    current_depth = 0
    buffer = []
    for i, line in enumerate(map(lambda line: line.strip(), file_to_read)):
        if current_depth == 0 and line in ("[", "]"):
            continue

        if line == "{":
            current_depth += 1

        if line == "}" or line == "},":
            current_depth -= 1
            if current_depth == 0:
                buffer.append("}")
                yield json.loads(" ".join(buffer))
                buffer = []
            else:
                buffer.append(line)
        else:
            buffer.append(line)
