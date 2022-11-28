from LSTMTester import LSTMTester
from INNTester import INNTester

def run_tests():
    # lstm_tester = LSTMTester()

    # lstm_tester.set_input_data(30, 30, "offset")

    # lstm_tester.get_data()


    inn_tester = INNTester()

    inn_tester.set_input_data(30, 30, "offset")

    inn_tester.get_data()

if __name__ == "__main__":
    run_tests()
