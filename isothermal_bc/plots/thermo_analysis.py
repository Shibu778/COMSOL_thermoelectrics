# This script performs the Power and Seebeck coefficient analysis of the thermoelectric data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib as mpl
from pathlib import Path
import os

## Use style file
style_file = Path(__file__).parent / "mystyle.mplstyle"
style.use(style_file)
plt.rcParams["lines.markersize"] = 5

markers = {"S": "o-", "HC": "s-", "HH": "h-", "HM": "<-"}
colors = {"S": "#f0a3ff", "HC": "#0075dc", "HH": "#993f00", "HM": "#4c005c"}


class CHF_Experiment:
    # A class to perform the Power and Seebeck coefficient analysis of the thermoelectric data
    # when the heat flux at the bottom plate is constant
    # CHF: Constant Heat Flux
    def __init__(
        self,
        data_path="../data/power/",
        system="S",
        save_path="./S",
        save_plotdata=True,
        plot_format="png",
    ):
        self.path_vhp = os.path.join(data_path, system + "_VHP_vs_THP_vs_I0.txt")
        self.path_tcp = os.path.join(data_path, system + "_TCP_vs_THP_vs_I0.txt")
        self.heat_flux = 5  # W/(m^2K)
        self.vhp_data = self.read_vhp_vs_thp_vs_i0()
        self.tcp_data = self.read_tcp_vs_thp_vs_i0()
        self.tdep_power_data = self.tdep_power_simulation(self.vhp_data)
        self.seebeck_data = self.seebeck_simulation(self.tcp_data, self.tdep_power_data)
        self.save_path = save_path
        self.system = system
        self.save_plotdata = save_plotdata
        self.plot_format = plot_format

    def plot_temp_diff_vs_thps_with_current(self):
        # Plots temperature difference vs Hot Plate Temperature with current
        thps = self.tdep_power_data["THP"]
        tcps = {}
        data = self.tcp_data["TCP_data"]
        for i in [0.0, 0.5, 1.0, 1.5, 2.0]:
            tcps[i] = []
        for key in data.keys():
            for it in data[key]:
                tcps[it[0]].append(it[1])

        fig, ax = plt.subplots(figsize=(5, 4))
        for key in tcps:
            temp_diff = np.array(thps) - np.array(tcps[key])
            plt.plot(thps, temp_diff, markers[self.system], label=f"I={key} A")
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Temperature Difference (K)")
        # plt.title(f"Temperature Difference vs Hot Plate Temperature with Current")
        # plt.grid()
        # make colorbar legend
        legend = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fancybox=True,
            shadow=True,
        )
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path,
            "Temp_Diff_vs_Hot_Plate_Temperature_with_Current." + self.plot_format,
        )
        plt.savefig(filename, dpi=300)

        # print(data)
        # print(tcps)
        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Temp_Diff_vs_Hot_Plate_Temperature_with_Current.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,")
                for key in tcps:
                    file.write(f"delT_at_I0={key}A,")
                file.write("\n")
                for i in range(len(thps)):
                    file.write(f"{thps[i]},")
                    for key in tcps:
                        file.write(f"{thps[i]-tcps[key][i]},")
                    file.write("\n")

        return tcps

    def plot_temp_diff_vs_thps(self):
        # Plots temperature difference vs Hot Plate Temperature
        thps = self.tdep_power_data["THP"]
        tcps = self.seebeck_data["TCP_I0=0"]
        temp_diff = np.array(thps) - np.array(tcps)
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(thps, temp_diff, markers[self.system], c="r")
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Temperature Difference (K)")
        # plt.title("Temperature Difference vs Hot Plate Temperature at I0=0A")
        plt.grid()
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path, "Temp_Diff_vs_Hot_Plate_Temperature." + self.plot_format
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Temp_Diff_vs_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,delT\n")
                for i in range(len(thps)):
                    file.write(f"{thps[i]},{temp_diff[i]}\n")

    def plot_max_power_and_seebeck_coefficient_vs_thps(self):
        # Plot max power and Seebeck Coefficient with Hot Plate Temperature
        tdep_power_data = self.tdep_power_data
        seebeck_data = self.seebeck_data
        thps = tdep_power_data["THP"]
        fig, ax1 = plt.subplots(figsize=(5, 4))
        color = "tab:red"
        ax1.set_xlabel("Hot Plate Temperature (K)")
        ax1.set_ylabel("Maximum Power ($\mu$W)", color=color)
        ax1.plot(
            thps,
            np.array(
                [data["max_power"] for data in tdep_power_data["electrical_properties"]]
            )
            * 1e06,
            markers[self.system],
            color=color,
        )
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Seebeck Coefficient ($\mu$V/K)", color=color)
        ax2.plot(
            thps,
            np.array(seebeck_data["seebeck_coefficient"]) * 1e06,
            markers[self.system],
            color=color,
        )
        ax2.tick_params(axis="y", labelcolor=color)
        fig.tight_layout()
        filename = os.path.join(
            self.save_path,
            "Max_Power_and_Seebeck_Coefficient_vs_THP." + self.plot_format,
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Max_Power_and_Seebeck_Coefficient_vs_THP.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,Max_Power,Seebeck_Coefficient\n")
                for i in range(len(thps)):
                    file.write(
                        f"{thps[i]},{tdep_power_data['electrical_properties'][i]['max_power']},{seebeck_data['seebeck_coefficient'][i]}\n"
                    )

    def plot_max_power_vs_hot_plate_temperature(self):
        # Plot the maximum power vs hot plate temperature
        tdep_power_data = self.tdep_power_data
        thps = tdep_power_data["THP"]
        max_powers = [
            data["max_power"] for data in tdep_power_data["electrical_properties"]
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(thps, np.array(max_powers) * 1e06, markers[self.system], c="g")
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Maximum Power ($\mu$W)")
        # plt.title("Maximum Power vs Hot Plate Temperature")
        plt.grid()
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path, "Max_Power_vs_Hot_Plate_Temperature." + self.plot_format
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Max_Power_vs_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,Max_Power\n")
                for i in range(len(thps)):
                    file.write(f"{thps[i]},{max_powers[i]}\n")

    def plot_power_vs_I_with_hot_plate_temperature(self):
        # Plot the Power vs Current with hot plate temperature
        tdep_power_data = self.tdep_power_data
        thps = tdep_power_data["THP"]
        fig, ax = plt.subplots(figsize=(5, 4))
        i_max = max(
            [im["I_at_max_power"] for im in tdep_power_data["electrical_properties"]]
        )

        for i, thp in enumerate(thps):
            fit = tdep_power_data["electrical_properties"][i]["fit"]
            max_power = tdep_power_data["electrical_properties"][i]["max_power"]
            i_at_max_power = tdep_power_data["electrical_properties"][i][
                "I_at_max_power"
            ]
            I_range = np.linspace(0, i_max + 0.05, 100)
            V = np.polyval(fit, I_range)
            P = -I_range * V * 1e06
            plt.plot(I_range, P, label=f"{thp} K")
            plt.scatter(
                i_at_max_power, max_power * 1e06, c="k", marker=markers[self.system][:1]
            )
        plt.xlabel(r"Current (A)")
        plt.ylabel(r"Power ($\mu$W)")
        # plt.title("Power vs Current with Hot Plate Temperature")
        plt.grid()
        legend = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fancybox=True,
            shadow=True,
        )
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        fig.tight_layout()
        filename = os.path.join(
            self.save_path, "P_vs_I_with_Hot_Plate_Temperature." + self.plot_format
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "P_vs_I_with_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,I,P\n")
                for i, thp in enumerate(thps):
                    fit = tdep_power_data["electrical_properties"][i]["fit"]
                    max_power = tdep_power_data["electrical_properties"][i]["max_power"]
                    i_at_max_power = tdep_power_data["electrical_properties"][i][
                        "I_at_max_power"
                    ]
                    I_range = np.linspace(0, i_max + 0.05, 100)
                    V = np.polyval(fit, I_range)
                    P = -I_range * V * 1e06
                    for j in range(len(I_range)):
                        file.write(f"{thp},{I_range[j]},{P[j]}\n")

    def plot_V_vs_I_with_hot_plate_temperature(self):
        # Plot the Voltage vs Current with hot plate temperature
        tdep_power_data = self.tdep_power_data
        thps = tdep_power_data["THP"]
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, thp in enumerate(thps):
            I = np.array(tdep_power_data["data_VI"][i]["I"])
            V = np.array(tdep_power_data["data_VI"][i]["V"]) * 1e03
            plt.plot(I, V, label=f"{thp} K")
        plt.xlabel(r"Current (A)")
        plt.ylabel(r"Voltage (mV)")
        x_max = max(
            [
                im["short_circuit_current"]
                for im in tdep_power_data["electrical_properties"]
            ]
        )
        y_min = min(
            [
                im["open_circuit_voltage"]
                for im in tdep_power_data["electrical_properties"]
            ]
        )
        plt.xlim(0, x_max + 0.2)
        plt.ylim(y_min * 1e03 - 2, 4)
        # plt.title("Voltage vs Current with Hot Plate Temperature")
        plt.grid()
        # make colorbar legend
        # plt.legend()
        legend = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fancybox=True,
            shadow=True,
        )
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        fig.tight_layout()
        filename = os.path.join(
            self.save_path, "V_vs_I_with_Hot_Plate_Temperature." + self.plot_format
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "V_vs_I_with_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,I,V\n")
                for i, thp in enumerate(thps):
                    I = np.array(tdep_power_data["data_VI"][i]["I"])
                    V = np.array(tdep_power_data["data_VI"][i]["V"]) * 1e03
                    for j in range(len(I)):
                        file.write(f"{thp},{I[j]},{V[j]}\n")

    def plot_seebeck_coefficient(self):
        # Plot the Seebeck coefficient vs Hot Plate Temperature
        seebeck_data = self.seebeck_data
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(
            np.array(seebeck_data["THP"]),
            np.array(seebeck_data["seebeck_coefficient"]) * 1e06,
            markers[self.system],
            c="b",
        )
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Seebeck Coefficient ($\mu$V/K)")
        # plt.title("Seebeck Coefficient vs Hot Plate Temperature")
        plt.grid()
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path,
            "Seebeck_Coefficient_vs_Hot_Plate_Temperature." + self.plot_format,
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Seebeck_Coefficient_vs_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,Seebeck_Coefficient\n")
                for i in range(len(seebeck_data["THP"])):
                    file.write(
                        f"{seebeck_data['THP'][i]},{seebeck_data['seebeck_coefficient'][i]}\n"
                    )

    def read_vhp_vs_thp_vs_i0(self):
        # Read the VHP vs THP vs I0 data
        with open(self.path_vhp, "r") as file:
            data1 = file.readlines()
        data1 = [line.strip() for line in data1]
        data = {}
        data["Model"] = data1[0].split()[2]
        data["Version"] = " ".join(data1[1].split()[2:])
        data["Time"] = " ".join(data1[2].split()[2:])
        data["Dimension"] = data1[3].split()[2]
        data["Nodes"] = data1[4].split()[2]
        data["Expressions"] = data1[5].split()[2]
        data["Descriptions"] = " ".join(data1[6].split()[2:])
        data["Length_unit"] = data1[7].split()[-1]
        headers = data1[8].split("V (V) @")
        values = data1[9].split()
        values = [float(value) for value in values]
        headers = [header.strip().split(", ") for header in headers][1:]
        data["measurement_point_xyz"] = values[:3]
        data["VHP_headers"] = headers
        # data["VHP_headers"] = [
        #     {
        #         header[0].split("=")[0]: float(header[0].split("=")[1]),
        #         header[1].split("=")[0]: float(header[1].split("=")[1]),
        #     }
        #     for header in headers
        # ]
        data["VHP_values"] = values[3:]
        # Storing data in grids
        length = len(data["VHP_values"])
        vhp_data = {}
        for i in range(length):
            vhp_data[headers[i][1]] = []
        for i in range(length):
            vhp_data[headers[i][1]].append(
                [float(headers[i][0].split("=")[1]), data["VHP_values"][i]]
            )

        data["VHP_data"] = vhp_data
        data["VHP_description"] = (
            """VHP_data is a dictionary with the key representing the Hot Plate Temperature and the value contains a list of [I0, VHP],  where I0 is the current applied at the Hot Plate and VHP is the voltage measured at the Hot Plate."""
        )
        return data

    def read_tcp_vs_thp_vs_i0(self):
        # Read the VHP vs THP vs I0 data
        with open(self.path_tcp, "r") as file:
            data1 = file.readlines()
        data1 = [line.strip() for line in data1]
        data = {}
        data["Model"] = data1[0].split()[2]
        data["Version"] = " ".join(data1[1].split()[2:])
        data["Time"] = " ".join(data1[2].split()[2:])
        data["Dimension"] = data1[3].split()[2]
        data["Nodes"] = data1[4].split()[2]
        data["Expressions"] = data1[5].split()[2]
        data["Descriptions"] = " ".join(data1[6].split()[2:])
        data["Length_unit"] = data1[7].split()[-1]
        headers = data1[8].split("T (K) @")
        values = data1[9].split()
        values = [float(value) for value in values]
        headers = [header.strip().split(", ") for header in headers][1:]
        data["measurement_point_xyz"] = values[:3]
        data["TCP_headers"] = headers
        # data["VHP_headers"] = [
        #     {
        #         header[0].split("=")[0]: float(header[0].split("=")[1]),
        #         header[1].split("=")[0]: float(header[1].split("=")[1]),
        #     }
        #     for header in headers
        # ]
        data["TCP_values"] = values[3:]
        # Storing data in grids
        length = len(data["TCP_values"])
        tcp_data = {}
        for i in range(length):
            tcp_data[headers[i][1]] = []
        for i in range(length):
            tcp_data[headers[i][1]].append(
                [float(headers[i][0].split("=")[1]), data["TCP_values"][i]]
            )

        data["TCP_data"] = tcp_data
        data["TCP_description"] = (
            """TCP_data is a dictionary with the key representing the Hot Plate Temperature and the value contains a list of [I0, TCP],  where I0 is the current applied at the Hot Plate and TCP is the temperature measured at the Cold Plate."""
        )
        return data

    def tdep_power_simulation(self, vhp_data):
        """This function calculates the temperature dependent relevant data from the VHP vs THP vs I0 data"""
        data = vhp_data["VHP_data"]
        tdep_relevant_data = {"THP": [], "data_VI": [], "electrical_properties": []}
        for key in data.keys():
            data_VI = {"I": [], "V": []}
            for i in range(len(data[key])):
                data_VI["I"].append(data[key][i][0])
                data_VI["V"].append(data[key][i][1])
            tdep_relevant_data["THP"].append(float(key.split("=")[1]))
            tdep_relevant_data["data_VI"].append(data_VI)
            tdep_relevant_data["electrical_properties"].append(
                self.calc_relevant_data(data_VI)
            )
        return tdep_relevant_data

    def seebeck_simulation(self, tcp_data, tdep_power_data):
        """Finds the Seebeck coefficient from the TCP vs THP when I0 = 0 A data.
        Here, we assume that the first data point is when I0 = 0 A"""

        data = tcp_data["TCP_data"]
        seebeck_data = {
            "THP": [],
            "TCP_I0=0": [],
            "VHP_I0=0": [],
            "seebeck_coefficient": [],
        }
        for key in data.keys():
            seebeck_data["THP"].append(float(key.split("=")[1]))
            seebeck_data["TCP_I0=0"].append(data[key][0][1])
        # Check whether the THP values are the same in both the data
        assert seebeck_data["THP"] == tdep_power_data["THP"]
        for i in range(len(seebeck_data["THP"])):
            seebeck_data["VHP_I0=0"].append(tdep_power_data["data_VI"][i]["V"][0])
            seebeck_data["seebeck_coefficient"].append(
                tdep_power_data["data_VI"][i]["V"][0]
                / (seebeck_data["THP"][i] - seebeck_data["TCP_I0=0"][i])
            )
        return seebeck_data

    def find_linear_fit(self, x, y):
        # Find the linear fit of the data
        fit = np.polyfit(x, y, 1)
        return fit

    def calc_relevant_data(self, data_VI):
        # Calculate the relevant data from the V vs I data
        fit = self.find_linear_fit(data_VI["I"], data_VI["V"])
        max_power = self.calc_max_power(fit)
        I_at_max_power = self.calc_I_at_max_power(fit)
        V_at_max_power = np.polyval(fit, I_at_max_power)
        open_circuit_voltage = self.calc_open_circuit_voltage(fit)
        short_circuit_current = self.calc_short_circuit_current(fit)
        return {
            "fit": fit,
            "max_power": max_power,
            "I_at_max_power": I_at_max_power,
            "V_at_max_power": V_at_max_power,
            "open_circuit_voltage": open_circuit_voltage,
            "short_circuit_current": short_circuit_current,
        }

    def calc_I_at_max_power(self, fit):
        # Calculate the current at which the maximum power is obtained
        m, b = fit
        return -b / (2 * m)

    def calc_max_power(self, fit):
        # Calculate the maximum power from the linear fit
        m, b = fit
        return b**2 / (4 * m)

    def calc_open_circuit_voltage(self, fit):
        # Calculate the open circuit voltage from the linear fit
        _, b = fit
        return b

    def calc_short_circuit_current(self, fit):
        # Calculate the short circuit current from the linear fit
        m, b = fit
        return -b / m


class Compare_Geometry:
    def __init__(
        self,
        data_path="../data/power/",
        systems=["S", "HC", "HH", "HM"],
        save_path="./compare/",
        save_plotdata=True,
        plot_format="png",
    ):
        self.data_path = data_path
        self.systems = systems
        self.save_path = save_path
        self.chf_data = {}
        self.save_plotdata = save_plotdata
        self.plot_format = plot_format
        for system in systems:
            self.chf_data[system] = CHF_Experiment(
                data_path=data_path, system=system, save_path=f"./{system}"
            )

    def plot_max_power_and_seebeck_coefficient_vs_thps(self):
        # Plot max power and Seebeck Coefficient with Hot Plate Temperature
        fig, ax1 = plt.subplots(figsize=(5, 4))
        # markers = {"S": "o-", "HC": "s-", "HH": "h-", "HM": "<-"}
        # marker_size = 5
        # # colors = {"S": "#f0a3ff", "HC": "#0075dc", "HH": "#993f00", "HM": "#4c005c"}
        for i, system in enumerate(self.systems):
            tdep_power_data = self.chf_data[system].tdep_power_data
            seebeck_data = self.chf_data[system].seebeck_data
            thps = tdep_power_data["THP"]
            color = "black"
            ax1.set_xlabel("Hot Plate Temperature (K)")
            ax1.set_ylabel("Maximum Power ($\mu$W)", color=color)
            ax1.plot(
                thps,
                np.array(
                    [
                        data["max_power"]
                        for data in tdep_power_data["electrical_properties"]
                    ]
                )
                * 1e06,
                markers[system],
                color=color,
                label=system,
            )
            ax1.tick_params(axis="y", labelcolor=color)
        ax2 = ax1.twinx()
        for i, system in enumerate(self.systems):
            tdep_power_data = self.chf_data[system].tdep_power_data
            seebeck_data = self.chf_data[system].seebeck_data
            thps = tdep_power_data["THP"]

            color = "tab:blue"
            ax2.set_ylabel("Seebeck Coefficient ($\mu$V/K)", color=color)
            ax2.plot(
                thps,
                np.array(seebeck_data["seebeck_coefficient"]) * 1e06,
                markers[system],
                color=color,
                label=system,
            )
            ax2.tick_params(axis="y", labelcolor=color)
        legend = plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.9),
            frameon=False,
            fancybox=True,
            shadow=True,
            ncol=2,  # Number of columns in the legend
        )
        fig.tight_layout()
        filename = os.path.join(
            self.save_path,
            "Max_Power_and_Seebeck_Coefficient_vs_THP." + self.plot_format,
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Max_Power_and_Seebeck_Coefficient_vs_THP.csv"
            )
            with open(filename, "w") as file:
                file.write(
                    "THP,Max_Power_S,Max_Power_HC,Max_Power_HH,Max_Power_HM,Seebeck_Coefficient_S,Seebeck_Coefficient_HC,Seebeck_Coefficient_HH,Seebeck_Coefficient_HM\n"
                )
                for i in range(len(thps)):
                    file.write(
                        f"{thps[i]},"
                        + ",".join(
                            [
                                str(
                                    self.chf_data[system].tdep_power_data[
                                        "electrical_properties"
                                    ][i]["max_power"]
                                )
                                for system in self.systems
                            ]
                        )
                        + ","
                        + ",".join(
                            [
                                str(
                                    self.chf_data[system].seebeck_data[
                                        "seebeck_coefficient"
                                    ][i]
                                )
                                for system in self.systems
                            ]
                        )
                        + "\n"
                    )

    def plot_temp_diff_vs_thps(self):
        # Plots temperature difference vs Hot Plate Temperature
        fig, ax = plt.subplots(figsize=(5, 4))
        for system in self.systems:
            thps = self.chf_data[system].tdep_power_data["THP"]
            tcps = self.chf_data[system].seebeck_data["TCP_I0=0"]
            temp_diff = np.array(thps) - np.array(tcps)
            plt.plot(thps, temp_diff, markers[system], label=system)
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Temperature Difference (K)")
        # plt.title("Temperature Difference vs Hot Plate Temperature")
        legend = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fancybox=True,
            shadow=True,
        )
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path, "Temp_Diff_vs_Hot_Plate_Temperature." + self.plot_format
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Temp_Diff_vs_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,delT_S,delT_HC,delT_HH,delT_HM\n")
                for i in range(len(thps)):
                    file.write(
                        f"{thps[i]},"
                        + ",".join(
                            [
                                str(
                                    self.chf_data[system].tdep_power_data["THP"][i]
                                    - self.chf_data[system].seebeck_data["TCP_I0=0"][i]
                                )
                                for system in self.systems
                            ]
                        )
                        + "\n"
                    )

    def plot_seebeck_coefficient(self):
        # Plot the Seebeck coefficient vs Hot Plate Temperature
        fig, ax = plt.subplots(figsize=(5, 4))
        for system in self.systems:
            seebeck_data = self.chf_data[system].seebeck_data
            plt.plot(
                np.array(seebeck_data["THP"]),
                np.array(seebeck_data["seebeck_coefficient"]) * 1e06,
                markers[system],
                label=system,
            )
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Seebeck Coefficient ($\mu$V/K)")
        # plt.title("Seebeck Coefficient vs Hot Plate Temperature")
        legend = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fancybox=True,
            shadow=True,
        )
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path,
            "Seebeck_Coefficient_vs_Hot_Plate_Temperature." + self.plot_format,
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Seebeck_Coefficient_vs_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write(
                    "THP,Seebeck_Coefficient_S,Seebeck_Coefficient_HC,Seebeck_Coefficient_HH,Seebeck_Coefficient_HM\n"
                )
                for i in range(len(seebeck_data["THP"])):
                    file.write(
                        f"{seebeck_data['THP'][i]},"
                        + ",".join(
                            [
                                str(
                                    self.chf_data[system].seebeck_data[
                                        "seebeck_coefficient"
                                    ][i]
                                )
                                for system in self.systems
                            ]
                        )
                        + "\n"
                    )

    def plot_max_power_vs_thps(self):
        # Plot the maximum power vs hot plate temperature
        fig, ax = plt.subplots(figsize=(5, 4))
        for system in self.systems:
            tdep_power_data = self.chf_data[system].tdep_power_data
            thps = tdep_power_data["THP"]
            max_powers = [
                data["max_power"] for data in tdep_power_data["electrical_properties"]
            ]
            plt.plot(thps, np.array(max_powers) * 1e06, markers[system], label=system)
        plt.xlabel(r"Hot Plate Temperature (K)")
        plt.ylabel(r"Maximum Power ($\mu$W)")
        # plt.title("Maximum Power vs Hot Plate Temperature")
        legend = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fancybox=True,
            shadow=True,
        )
        # Set grid properties
        ax.grid(which="major", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        filename = os.path.join(
            self.save_path, "Max_Power_vs_Hot_Plate_Temperature." + self.plot_format
        )
        plt.savefig(filename, dpi=300)

        if self.save_plotdata:
            filename = os.path.join(
                self.save_path, "Max_Power_vs_Hot_Plate_Temperature.csv"
            )
            with open(filename, "w") as file:
                file.write("THP,Max_Power_S,Max_Power_HC,Max_Power_HH,Max_Power_HM\n")
                for i in range(len(thps)):
                    file.write(
                        f"{thps[i]},"
                        + ",".join(
                            [
                                str(
                                    self.chf_data[system].tdep_power_data[
                                        "electrical_properties"
                                    ][i]["max_power"]
                                )
                                for system in self.systems
                            ]
                        )
                        + "\n"
                    )


# Main function
if __name__ == "__main__":
    # VHP vs THP vs I0 data
    for system in ["S", "HC", "HH", "HM"]:
        print("Calculating for system:", system)
        print("=====================================")
        chf = CHF_Experiment(system=system, save_path=f"./{system}", plot_format="svg")
        print(chf.vhp_data)
        print(chf.tcp_data)
        print(chf.tdep_power_data)
        print(chf.seebeck_data)
        chf.plot_seebeck_coefficient()
        chf.plot_V_vs_I_with_hot_plate_temperature()
        chf.plot_power_vs_I_with_hot_plate_temperature()
        chf.plot_max_power_vs_hot_plate_temperature()
        chf.plot_max_power_and_seebeck_coefficient_vs_thps()
        chf.plot_temp_diff_vs_thps()
        chf.plot_temp_diff_vs_thps_with_current()

    # Compare the geometry
    cg = Compare_Geometry(plot_format="svg")
    cg.plot_max_power_vs_thps()
    cg.plot_seebeck_coefficient()
    cg.plot_temp_diff_vs_thps()
    cg.plot_max_power_and_seebeck_coefficient_vs_thps()
