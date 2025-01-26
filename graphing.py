import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as col
import matplotlib.ticker as ticker
import matplotlib.colors as mc
from datetime import timedelta
from datetime import date
import pandas as pd
import numpy as np
import copy
import sys
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

from utility import *


class Grapher:

    def __init__(self, parameters):

        self.figure_width = parameters.get("figure_width", 7.5)
        self.figure_height = parameters.get("figure_height", 4.5)
        self.map_width = parameters.get("map_width", 9)
        self.map_height = parameters.get("map_height", 5)

        self.x_multiplier = parameters.get("x_multiplier", None)

        self.x_label = parameters.get("x_label", "")
        self.y_label = parameters.get("y_label", "")
        self.x_label_weight = parameters.get("x_label_weight", "normal")
        self.y_label_weight = parameters.get("y_label_weight", "normal")
        self.x_label_color = parameters.get("x_label_color", "k")
        self.y_label_color = parameters.get("y_label_color", "k")
        self.x_tick_label_weight = parameters.get("x_tick_label_weight", "normal")
        self.y_tick_label_weight = parameters.get("ytick_label_weight", "normal")
        self.x_scale = parameters.get("x_scale", None)
        self.y_scale = parameters.get("y_scale", None)
        self.x_min = parameters.get("x_min", 0)
        self.x_max = parameters.get("x_max", None)
        self.x_tick = parameters.get("x_tick", None)
        self.x_ticks = parameters.get("x_ticks", None)
        self.x_tick_labels = parameters.get("x_tick_labels", None)
        self.x_tick_label_colors = parameters.get("x_tick_label_colors", None)
        self.x_tick_label_weight = parameters.get("x_tick_label_weight", "normal")
        self.y_min = parameters.get("y_min", 0)
        self.y_max = parameters.get("y_max", None)
        self.y_tick = parameters.get("y_tick", None)
        self.y_ticks = parameters.get("y_ticks", None)
        self.y_tick_labels = parameters.get("y_tick_labels", None)
        self.y_tick_label_colors = parameters.get("y_tick_label_colors", None)
        self.y_tick_label_weight = parameters.get("y_tick_label_weight", "normal")

        self.alternate_axis = parameters.get("alternate_axis", None)
        self.alternate_x_label = parameters.get("alternate_x_label", "")
        self.alternate_y_label = parameters.get("alternate_y_label", "")
        self.alternate_x_label_weight = parameters.get("alternate_x_label_weight", "normal")
        self.alternate_y_label_weight = parameters.get("alternate_y_label_weight", "normal")
        self.alternate_x_label_color = parameters.get("alternate_x_label_color", "k")
        self.alternate_y_label_color = parameters.get("alternate_y_label_color", "k")
        self.alternate_x_scale = parameters.get("alternate_x_scale", None)
        self.alternate_y_scale = parameters.get("alternate_y_scale", None)
        self.alternate_x_min = parameters.get("alternate_x_min", 0)
        self.alternate_x_max = parameters.get("alternate_x_max", None)
        self.alternate_x_tick = parameters.get("alternate_x_tick", None)
        self.alternate_x_ticks = parameters.get("alternate_x_ticks", None)
        self.alternate_x_tick_labels = parameters.get("alternate_x_tick_labels", None)
        self.alternate_x_tick_label_colors = parameters.get("alternate_x_tick_label_colors", None)
        self.alternate_x_tick_label_weight = parameters.get("alternate_x_tick_label_weight", "normal")
        self.alternate_y_min = parameters.get("alternate_y_min", 0)
        self.alternate_y_max = parameters.get("alternate_y_max", None)
        self.alternate_y_tick = parameters.get("alternate_y_tick", None)
        self.alternate_y_ticks = parameters.get("alternate_y_ticks", None)
        self.alternate_y_tick_labels = parameters.get("alternate_y_tick_labels", None)
        self.alternate_y_tick_label_colors = parameters.get("alternate_y_tick_label_colors", None)
        self.alternate_y_tick_label_weight = parameters.get("alternate_y_tick_label_weight", "normal")

        self.bar_width = parameters.get("bar_width", 0.2)
        self.bar_colors = parameters.get("bar_colors", plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.sub_bar_colors = parameters.get("sub_bar_colors", plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.line_colors = parameters.get("line_colors", plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.alternate_line_colors = parameters.get("alternate_line_colors", plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.average_line_colors = parameters.get("average_line_colors", plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.span_colors = parameters.get("span_colors", plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.line_styles = parameters.get("line_styles", None)
        self.line_formats = parameters.get("line_formats", None)
        self.line_widths = parameters.get("line_widths", None)
        self.alternate_line_styles = parameters.get("alternate_line_styles", None)
        self.average_line_styles = parameters.get("average_line_styles", None)
        self.legend = parameters.get("legend", True)
        self.legend_location = parameters.get("legend_location", "upper right")
        self.alternate_legend_location = parameters.get("alternate_legend_location", "lower right")

        self.line_name_translator = parameters.get("line_name_translator", None)
        self.alternate_line_name_translator = parameters.get("alternate_line_name_translator", None)
        self.bar_name_translator = parameters.get("bar_name_translator", None)
        self.group_name_translator = parameters.get("group_name_translator", None)
        self.sub_bar_name_translator = parameters.get("sub_bar_name_translator", None)
        self.alternate_line_name_translator = parameters.get("alternate_line_name_translator", None)

        self.map_vmax = parameters.get("map_vmax", 10e5)
        self.map_bins = parameters.get("map_bins", (300, 300))
        self.map_cmap = parameters.get("map_cmap", "hot")

    def graph_setup(self, ax1):
        ax1.set_xlabel(self.x_label, weight=self.x_label_weight)
        ax1.xaxis.label.set_color(self.x_label_color)
        ax1.set_ylabel(self.y_label, weight=self.y_label_weight)
        ax1.yaxis.label.set_color(self.y_label_color)
        if self.x_scale is not None:
            ax1.set_xscale(self.x_scale)
        if self.y_scale is not None:
            ax1.set_yscale(self.y_scale)
        if self.x_max is not None:
            ax1.set_xlim(xmin=self.x_min, xmax=self.x_max)
        if self.x_tick is not None:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(self.x_tick))
        if self.x_ticks is not None:
            ax1.set_xticks(self.x_ticks)
            ax1.set_xticklabels(self.x_tick_labels)
        if self.x_tick_label_colors is not None:
            x_tick_label_count = len(ax1.get_xticklabels())
            for i in range(x_tick_label_count):
                ax1.get_xticklabels()[i].set_color(self.x_tick_label_colors[i])
        if self.y_max is not None:
            ax1.set_ylim(ymin=self.y_min, ymax=self.y_max)
        if self.y_tick is not None:
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(self.y_tick))
        if self.y_ticks is not None:
            ax1.set_yticks(self.y_ticks)
            ax1.set_yticklabels(self.y_tick_labels)
        if self.y_tick_label_colors is not None:
            y_tick_label_count = len(ax1.get_yticklabels())
            for i in range(y_tick_label_count):
                ax1.get_yticklabels()[i].set_color(self.y_tick_label_colors[i])
        ax2 = None
        if self.alternate_axis is not None:
            if self.alternate_axis == "x":
                ax2 = ax1.twiny()
            if self.alternate_axis == "y":
                ax2 = ax1.twinx()
        if ax2 is not None:
            if self.alternate_axis == "x":
                ax2.set_xlabel(self.alternate_x_label, weight=self.alternate_x_label_weight)
                ax2.xaxis.label.set_color(self.alternate_x_label_color)
                if self.alternate_x_scale is not None:
                    ax2.set_xscale(self.alternate_x_scale)
                if self.alternate_x_max is not None:
                    ax2.set_xlim(xmin=self.alternate_x_min, xmax=self.alternate_x_max)
                if self.alternate_x_tick is not None:
                    ax2.xaxis.set_major_locator(ticker.MultipleLocator(self.alternate_x_tick))
                if self.alternate_x_ticks is not None:
                    ax2.set_xticks(self.alternate_x_ticks)
                    ax2.set_xticklabels(self.alternate_x_tick_labels)
                    ax2.set_xlim(xmin=self.alternate_x_min, xmax=self.alternate_x_max)
            if self.alternate_axis == "y":
                ax2.set_ylabel(self.alternate_y_label, weight=self.alternate_y_label_weight)
                ax2.yaxis.label.set_color(self.alternate_y_label_color)
                if self.alternate_y_scale is not None:
                    ax2.set_yscale(self.alternate_y_scale)
                if self.alternate_y_max is not None:
                    ax2.set_ylim(ymin=self.alternate_y_min, ymax=self.alternate_y_max)
                if self.alternate_y_tick is not None:
                    ax2.yaxis.set_major_locator(ticker.MultipleLocator(self.alternate_y_tick))
                if self.alternate_y_ticks is not None:
                    ax2.set_yticks(self.alternate_y_ticks)
                    ax2.set_yticklabels(self.alternate_y_tick_labels)
                    ax2.set_ylim(ymin=self.alternate_y_min, ymax=self.alternate_y_max)
        ax1.grid()

    def line_graph(self, line_name_list, x_line_list, y_line_list, include_average=False, span_lists=None):
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        if self.line_name_translator is not None:
            line_name_list_ = [self.line_name_translator[line_name] for line_name in line_name_list]
        else:
            line_name_list_ = line_name_list
        if self.line_formats is None:
            self.line_formats = [""] * len(line_name_list)
        if self.line_widths is None:
            self.line_widths = [1.5] * len(line_name_list)
        if self.line_styles is None:
            self.line_styles = ["-"] * len(line_name_list)
        if self.average_line_styles is None:
            self.average_line_styles = ["--"] * len(line_name_list)
        i = 0
        for x, y in zip(x_line_list, y_line_list):
            average = float(sum(y)) / len(x)
            ax1.plot(x, y, linewidth=self.line_widths[i], color=self.line_colors[i], linestyle=self.line_styles[i], solid_capstyle="butt", zorder=100 if i == 1 else 0)
            if include_average:
                ax1.hlines(y=average, xmin=0, xmax=max(x), linewidth=1.5, color=self.average_line_colors[i], linestyle=self.average_line_styles[i])
            i += 1
        i = 0
        if span_lists is not None:
            for span_list in span_lists:
                for span in span_list:
                    plt.axvspan(span[0], span[1], facecolor=self.span_colors[i], alpha=0.6)
                i += 1
        if self.legend:
            ax1.legend(line_name_list_, loc=self.legend_location)
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()

    def bar_graph(self, bar_name_list, y):
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        if self.bar_name_translator is not None:
            bar_name_list_ = [self.bar_name_translator[bar_name] for bar_name in bar_name_list]
        else:
            bar_name_list_ = bar_name_list
        x_pos_list = [float(x_) for x_ in list(range(len(bar_name_list)))]
        ax1.bar(x_pos_list, y, width=self.bar_width, color=self.bar_colors)
        ax1.set_xticks(x_pos_list)
        ax1.set_xticklabels(bar_name_list_, weight=self.x_tick_label_weight)
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()

    def sorted_line_graph(self, line_name_list, y_line_list, include_average=False):
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        if self.line_name_translator is not None:
            line_name_list_ = [self.line_name_translator[line_name] for line_name in line_name_list]
        else:
            line_name_list_ = line_name_list
        if self.line_styles is None:
            self.line_styles = ["-"] * len(line_name_list)
        if self.average_line_styles is None:
            self.average_line_styles = ["--"] * len(line_name_list)
        i = 0
        for y_line in y_line_list:
            x = list(range(1, len(y_line) + 1))
            y = sorted(y_line)
            average = float(sum(y)) / len(x)
            # [x[k] * self.x_multiplier[i] for k in range(len(x))]
            ax1.plot(x, y, linewidth=1.5, color=self.line_colors[i], linestyle=self.line_styles[i], label=line_name_list_[i])
            if include_average:
                ax1.hlines(y=average, xmin=0, xmax=max(x), linewidth=1.5, color=self.average_line_colors[i], linestyle=self.average_line_styles[i], label='_nolegend_')
            i += 1
        ax1.legend(line_name_list_, loc=self.legend_location)
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()

    from scipy.interpolate import UnivariateSpline
    import numpy as np

    def scatter_plot_graph(self, plot_name_list, y_plot_list, x_plot_list):
        window = 200
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        i = 0
        for y_plot, x_plot in zip(y_plot_list, x_plot_list):
            x_plot, y_plot = (list(t) for t in zip(*sorted(zip(x_plot, y_plot))))
            # average_y = []
            # for ind in range(len(y_plot) - window + 1):
            #     average_y.append(np.mean(y_plot[ind:ind + window]))
            # for ind in range(window - 1):
            #     average_y.insert(0, np.nan)
            average = float(sum(y_plot)) / len(x_plot)
            print(plot_name_list[i], average)
            ax1.hlines(y=average, xmin=0, xmax=self.x_max, linewidth=1.5, color=self.line_colors[i], linestyles="--", label='_nolegend_')
            ax1.scatter(x_plot, y_plot, s=3, c=self.line_colors[i])
            # ax1.plot(x_plot, average_y, color=self.line_colors[i], label=plot_name_list[i])
            i += 1
        ax1.legend(plot_name_list, loc=self.legend_location, markerscale=2.0)
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()


    def sorted_grouped_line_graph(self, line_name_list, alternate_line_name_list, y_line_dictionary_list):
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        if self.line_name_translator is not None:
            line_name_list_ = [self.line_name_translator[line_name] for line_name in line_name_list]
        else:
            line_name_list_ = line_name_list
        if self.alternate_line_name_translator is not None:
            alternate_line_name_list_ = [self.alternate_line_name_translator[alternate_line_name] for alternate_line_name in alternate_line_name_list]
        else:
            alternate_line_name_list_ = alternate_line_name_list
        i = 0
        for y_line_dictionary in y_line_dictionary_list:
            for alternate_line_name in y_line_dictionary:
                if alternate_line_name in y_line_dictionary:
                    y_line = y_line_dictionary[alternate_line_name]
                    x = list(range(1, len(y_line) + 1))
                    y = sorted(y_line)
                else:
                    x = []
                    y = []
                ax1.plot(x, y, linewidth=1.5, color=self.line_colors[i], linestyle=self.line_styles[alternate_line_name_list.index(alternate_line_name)])
            i += 1
        ax1.legend(line_name_list_, loc=self.legend_location)
        lines = ax1.get_lines()
        legend1 = plt.legend([lines[i] for i in [j * len(alternate_line_name_list) for j in range(len(line_name_list))]], line_name_list_, loc=self.legend_location)
        legend2 = plt.legend([lines[i] for i in range(len(alternate_line_name_list))], alternate_line_name_list_, loc=self.alternate_legend_location)
        ax1.add_artist(legend1)
        ax1.add_artist(legend2)
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()

    def bar_graph(self, bar_name_list, y):
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        if self.bar_name_translator is not None:
            bar_name_list_ = [self.bar_name_translator[bar_name] for bar_name in bar_name_list]
        else:
            bar_name_list_ = bar_name_list
        x_pos_list = [float(x_) for x_ in list(range(len(bar_name_list)))]
        ax1.bar(x_pos_list, y, width=self.bar_width, color=self.bar_colors)
        ax1.set_xticks(x_pos_list)
        ax1.set_xticklabels(bar_name_list_, weight=self.x_tick_label_weight)
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()

    def grouped_bar_graph(self, group_name_list, sub_bar_name_list, group_list):
        fig, ax1 = plt.subplots(figsize=(self.figure_width, self.figure_height))
        color_dictionary = {sub_bar_name: color for sub_bar_name, color in zip(sub_bar_name_list, self.sub_bar_colors)}
        if self.group_name_translator is not None:
            group_name_list_ = [self.group_name_translator[group_name] for group_name in group_name_list]
        else:
            group_name_list_ = group_name_list
        if self.sub_bar_name_translator is not None:
            sub_bar_name_list_ = [self.sub_bar_name_translator[sub_bar_name] for sub_bar_name in sub_bar_name_list]
        else:
            sub_bar_name_list_ = sub_bar_name_list
        x_list = [float(x) for x in list(range(len(group_name_list)))]
        x_centre_list = copy.deepcopy(x_list)
        i = 0
        for group in group_list:
            group_size = len(group)
            x_centre = x_centre_list[i]
            if group_size % 2 == 0:
                x_start = x_centre - self.bar_width * (group_size - 1) / 2.0
            else:
                x_start = x_centre - self.bar_width * int(group_size / 2)
            for bar in group:
                y = 0
                k = 0
                for sub_bar_name in bar:
                    y += bar[sub_bar_name]
                    ax1.bar([x_start], [y], width=self.bar_width, color=color_dictionary[sub_bar_name], zorder=len(bar) - 1 - k)
                    k += 1
                x_start += self.bar_width
            i += 1
        ax1.set_xticks(x_centre_list)
        ax1.set_xticklabels(group_name_list_, weight=self.x_tick_label_weight)
        ax1.legend(sub_bar_name_list_, loc=self.legend_location)
        legend = ax1.get_legend()
        for legend_i in range(len(sub_bar_name_list)):
            sub_bar_name = sub_bar_name_list[legend_i]
            legend.legendHandles[legend_i].set_color(color_dictionary[sub_bar_name])
        self.graph_setup(ax1)
        fig.tight_layout()
        plt.show()

    def frequency_distribution_line_graph(self, line_name_list, frequency_dictionary_list, include_average=False):

        y_line_list = []
        for frequency_dictionary in frequency_dictionary_list:
            y_line = []
            for frequency_key in frequency_dictionary:
                y_line.append(frequency_dictionary[frequency_key])
            y_line_list.append(y_line)

        self.sorted_line_graph(line_name_list, y_line_list, include_average=include_average)

    def grouped_frequency_distribution_line_graph(self, line_name_list, alternate_line_name_list, grouped_frequency_dictionary_list):

        y_line_dictionary_list = []
        for grouped_frequency_dictionary in grouped_frequency_dictionary_list:
            y_line_dictionary = {}
            for frequency_dictionary_key in grouped_frequency_dictionary:
                frequency_dictionary = grouped_frequency_dictionary[frequency_dictionary_key]
                y_line = []
                for frequency_key in frequency_dictionary:
                    y_line.append(frequency_dictionary[frequency_key])
                y_line_dictionary[frequency_dictionary_key] = y_line
            y_line_dictionary_list.append(y_line_dictionary)

        self.line_styles = ["-", "--", "-.", ":"]
        self.sorted_grouped_line_graph(line_name_list, alternate_line_name_list, y_line_dictionary_list)

    def geographical_density_map(self, location_lists):
        df = pd.DataFrame(location_lists[0], columns =['Longitude', 'Latitude'])

        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
        gdf = GeoDataFrame(df, geometry=geometry)

        # this is a simple map that goes with geopandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax = gdf.plot(ax=world.plot(figsize=(10, 6), color="grey", legend=True), markersize=0.5, cmap="plasma", legend=True)
        plt.axis('off')

        fig = ax.get_figure()
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=-1.0, vmax=1.0))
        sm._A = []
        fig.colorbar(sm, orientation='horizontal', fraction=0.075, pad=0.04)
        plt.show()




