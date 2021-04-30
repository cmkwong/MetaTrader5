
def print_dict(dict):
    print("\n~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    for key, value in dict.items():
        print("{}:\t{:.5f}".format(key, value))