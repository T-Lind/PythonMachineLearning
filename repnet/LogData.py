import numpy as np
import matplotlib.pyplot as plt


def compile_data(network_data) -> dict:
    compiled_length_data = [len(x) for x in network_data]
    poly_funcs = [np.poly1d(x) for x in compiled_length_data]
    derivatives = [func.deriv() for func in poly_funcs]
    return {
        "mean": np.mean(compiled_length_data),
        "standard deviation": np.std(compiled_length_data),
        "variance": np.var(compiled_length_data),
        "max": np.max(compiled_length_data),
        "min": np.min(compiled_length_data),
        "max rate of change": np.max(derivatives),
        "min rate of change": np.min(derivatives),
        "median": np.median(compiled_length_data),
        "first quartile": np.quantile(compiled_length_data, 0.25),
        "second quartile": np.quantile(compiled_length_data, 0.50),
        "third quartile": np.quantile(compiled_length_data, 0.75),
        "fourth quartile": np.quantile(compiled_length_data, 1),
    }


def format_data(data_dict: dict,
                variance=False, max_roc=False, min_roc=False, median=False, q1=False, q2=False, q3=False,
                q4=False) -> str:
    ret_str = \
        f"Data analysis for {f'{data_dict=}'.split('=')[0]}\n" \
        f"---------------------------\n" \
        f"Mean: {data_dict['mean']}\n" \
        f"Standard Deviation: {data_dict['standard deviation']}\n" \
        f"Minimum: {data_dict['min']}\n" \
        f"Maximum: {data_dict['max']}\n"

    if variance: ret_str += f"Variance: {data_dict['variance']}\n"
    if max_roc: ret_str += f"Max ROC: {data_dict['max rate of change']}"
    if min_roc: ret_str += f"Min ROC: {data_dict['min rate of change']}"
    if median: ret_str += f"Median:{data_dict['median']}"
    if q1: ret_str += f"Q1: {q1}"
    if q2: ret_str += f"Q2: {q2}"
    if q3: ret_str += f"Q3: {q3}"
    if q4: ret_str += f"Q4: {q4}"

    return ret_str


def print_compile_data(network_data) -> None:
    print(format_data(compile_data(network_data)))


def plot_data(network_data, label="Data", color="red", line_type="") -> None:
    for i in range(len(network_data)):
        data = network_data[i]

        if i == 0:
            plt.plot(data, line_type, label=label, color=color)
        else:
            plt.plot(data, line_type, color=color)
