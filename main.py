from ESN.ESN import ESN
from MackeyGlass.MackeyGlassGenerator import run

if __name__ == "__main__":
    data = run(1000)
    data_train = data[:700]
    data_test = data[700:]
    # print(data)
    esn = ESN(input_size=1, output_size=1, reservoir_size=100)
    esn.train(data_train)

    y_pred = esn.predict(data_test)

    print("Mean L1 Error: {}".format(esn.mean_l1_error(data_test[:-1], y_pred[1:])))
