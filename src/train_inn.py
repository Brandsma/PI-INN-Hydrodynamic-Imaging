from INN.main import simple_run
from INN.data import DataType

def train_inn(config=None, use_pde=False):
    simple_run(DataType.Hydro, subset="all", use_pde=use_pde, config=config)

if __name__ == '__main__':
    train_inn()
