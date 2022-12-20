from INN.main import simple_run
from INN.data import DataType

def main():
    simple_run(DataType.Hydro, subset="all", use_pde=True)


if __name__ == '__main__':
    main()
