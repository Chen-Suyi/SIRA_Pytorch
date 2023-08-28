import glob
import json
import os
import time

from prettytable import PrettyTable


def monitor():
    total_table = []
    for dataset in ["val", "test"]:
        table = PrettyTable([
            "exp_name", "exp_id", "enc_config", "reg_config", "loss_type",
            "R_RMSE", "R_MAE", "t_RMSE", "t_MAE", "Err_R", "Err_t", "score"
        ],
                            sortby="score",
                            header_style="upper",
                            valign="m",
                            title="{} Result".format(dataset),
                            reversesort=False)

        valid_dirs = []
        exp_dirs = glob.glob("../experiments/*/exp_*")
        # exp_dirs = [i for i in exp_dirs if i.split("/")[-2] in valid_dirs]

        for exp_dir in exp_dirs:
            params_json_path = os.path.join(exp_dir, "params.json")
            results_json_path = os.path.join(
                exp_dir, "{}_metrics_best.json".format(dataset))
            # logs_txt_path = os.path.join(exp_dir, "log.txt")
            if not os.path.exists(params_json_path) or not os.path.exists(
                    results_json_path):
                continue

            params = json.load(open(params_json_path, "r"))
            results = json.load(open(results_json_path, "r"))
            # exp info
            exp_name = exp_dir.split("/")[-2]
            exp_id = exp_dir.split("_")[-1]
            # results
            if "enc_config" in params:
                enc_config = str(params["enc_config"])
            else:
                enc_config = "-"
            if "fus_config" in params:
                fus_config = str(params["fus_config"])
            else:
                enc_config = "-"
            if "reg_config" in params:
                reg_config = str(params["reg_config"])
            else:
                reg_config = "-"
            loss_type = params["loss_type"]
            R_RMSE = "{:>5.4f}".format(results["R_RMSE"])
            R_MAE = "{:>5.4f}".format(results["R_MAE"])
            t_RMSE = "{:>5.4f}".format(results["t_RMSE"])
            t_MAE = "{:>5.4f}".format(results["t_MAE"])
            Err_R = "{:>5.4f}".format(results["Err_R"])
            Err_t = "{:>5.4f}".format(results["Err_t"])
            # CD = "{:>5.4f}".format(results["CD"])
            score = "{:>5.4f}".format(results["Err_R"] * 0.01 +
                                      results["Err_t"])

            cur_row = [
                exp_name, exp_id, enc_config, reg_config, loss_type, R_RMSE,
                R_MAE, t_RMSE, t_MAE, Err_R, Err_t, score
            ]
            table.add_row(cur_row)
        print(table)
        total_table.append(str(table))


def find():
    total_table = []
    for dataset in ["val", "test"]:
        table = PrettyTable([
            "exp_name", "exp_id", "R_RMSE", "R_MAE", "t_RMSE", "t_MAE",
            "Err_R", "Err_t", "score"
        ],
                            sortby="score",
                            header_style="upper",
                            valign="m",
                            title="{} Result".format(dataset),
                            reversesort=False)

        exp_dirs = glob.glob("../experiments/*/exp_*")

        for exp_dir in exp_dirs:
            params_json_path = os.path.join(exp_dir, "params.json")
            results_json_path = os.path.join(
                exp_dir, "{}_metrics_best.json".format(dataset))
            # logs_txt_path = os.path.join(exp_dir, "log.txt")
            if not os.path.exists(params_json_path) or not os.path.exists(
                    results_json_path):
                continue

            params = json.load(open(params_json_path, "r"))
            results = json.load(open(results_json_path, "r"))
            # exp info
            exp_name = exp_dir.split("/")[-2]
            exp_id = exp_dir.split("_")[-1]
            # results
            if "R_RMSE" not in results:
                continue
            R_RMSE = "{:>5.4f}".format(results["R_RMSE"])
            R_MAE = "{:>5.4f}".format(results["R_MAE"])
            t_RMSE = "{:>5.4f}".format(results["t_RMSE"])
            t_MAE = "{:>5.4f}".format(results["t_MAE"])
            Err_R = "{:>5.4f}".format(results["Err_R"])
            Err_t = "{:>5.4f}".format(results["Err_t"])
            score = "{:>5.4f}".format(results["Err_R"] * 0.01 +
                                      results["Err_t"])

            cur_row = [
                exp_name, exp_id, R_RMSE, R_MAE, t_RMSE, t_MAE, Err_R, Err_t,
                score
            ]
            table.add_row(cur_row)
        print(table)
        total_table.append(str(table))


def run(interval):
    while True:
        monitor()
        time.sleep(interval)


if __name__ == "__main__":
    # interval = 10 * 60
    # run(interval)

    find()
