def convert_str_to_bool(file):
    for key, value in file.items():
        if value == "True":
            file[key] = True
        if value == 'False':
            file[key] = False
    return file
