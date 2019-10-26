

def reverse_string_recursion(string):

    if len(string) <= 1:
        return string
    string[0], string[-1] = string[-1], string[0]
    string[1:-1] = reverse_string_recursion(string[1:-1])
    return string

if __name__ == "__main__":
    string1 = list('I am reversed')
    reverse_string_recursion(string1)
