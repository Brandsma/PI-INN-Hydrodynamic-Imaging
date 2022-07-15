import argparse


def is_int(n):
    try:
        float_n = float(n)
        int_n = int(float_n)
    except ValueError:
        return False
    else:
        return float_n == int_n


def is_float(n):
    try:
        _ = float(n)
    except ValueError:
        return False
    else:
        return True


def is_boolean(n):
    return n == "True" or n == "False"


def coords(s):
    """coords type for use in argparse. takes x,y as input and returns tuple"""
    try:
        x, y = map(int, s.split(' , '))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be 'x,y' and int")
