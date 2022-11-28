from INN.main import simple_run
from INN.data import DataType

def main():
    simple_run(DataType.Hydro, use_pde=False)


if __name__ == '__main__':
    main()
