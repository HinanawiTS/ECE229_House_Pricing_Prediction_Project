from bokeh.palettes import GnBu
from bokeh.transform import cumsum
import matplotlib.pyplot as plt
from bokeh.palettes import Blues
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure, ColumnDataSource
from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import Category10, Category20, Category20b, Category20c
from bokeh.palettes import PRGn, RdYlGn
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.layouts import row, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
from bokeh.embed import components
from bokeh.resources import CDN
import pandas as pd
from io import BytesIO
import base64

# df_new = pd.read_csv('final_dataframe.csv', index_col = 0)

from bokeh.models import WheelZoomTool
from bokeh.palettes import RdBu



def visualize_count(filtered_dataset):
    fs = filtered_dataset.copy()

    def wgs84_to_web_mercator(df, lon="longitude", lat="latitude"):
        k = 6378137
        df["x"] = df[lon] * (k * np.pi / 180.0)
        df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
        #df["price"] = df[lon] * 0
        return df

    CDMXhex = wgs84_to_web_mercator(fs)
    x = CDMXhex['x']
    y = CDMXhex['y']
    tile_provider = get_provider(Vendors.OSM)

    palette = list(reversed(Blues[7]))

    title = "Number of Airbnb Listings in "
    if fs["neighbourhood"].nunique() == 1:
        title = title + fs["neighbourhood"].iloc[0] + ", "
    if fs["neighbourhood_group"].nunique() < 5:
        for i in fs["neighbourhood_group"].unique():
            title = title + i + ", "
    title = title + "NYC"

    p = figure(title=title, match_aspect = False, tools = ["pan", "reset", "save", "wheel_zoom"], 
               x_axis_type="mercator", y_axis_type="mercator",plot_width=1050,plot_height=600)

    p.grid.visible = True

    sz = 1000
    if fs["neighbourhood_group"].nunique() == 1:
        sz = 500
    if fs["neighbourhood"].nunique() == 1:
        sz = 127
    r, bins = p.hexbin(x, y, size=sz,
                       line_color="white", line_alpha=0.2,
                       palette=palette, hover_color="pink", alpha=0.7, hover_alpha=0.2)
    p.add_tools(HoverTool(
        tooltips=[("Count", "@c")],
        show_arrow=True, mode="mouse", point_policy="follow_mouse", renderers=[r]))

    r = p.add_tile(tile_provider)
    r.level = "underlay"
    color_mapper = linear_cmap(field_name='price', palette=palette,
                               low=bins["counts"].min(), high=bins["counts"].max() + 1)
    color_bar = ColorBar(color_mapper=color_mapper['transform'],
                         formatter=NumeralTickFormatter(format='0.0[0000]'),
                         label_standoff=13, width=8, location=(0, 0), title="count")
    p.add_layout(color_bar, 'right')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    p.title.align = "right"
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.toolbar.logo = None
    script1, div1 = components(p)
    cdn_js = CDN.js_files[0]
    return script1, div1, cdn_js

def donut(dataset):
    fs = dataset.copy()
    # change figsize, must be two same numbers since it's square.
    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw=dict(aspect="equal"))
    column_name = fs.columns[0]
    recipe = (fs.groupby("room_type").count()[
              column_name] / len(fs)).reset_index()["room_type"]
    data = (fs.groupby("room_type").count()[
            column_name] / len(fs)).reset_index()[column_name]
    wedges, texts = ax.pie(data, wedgeprops=dict(
        width=0.5, edgecolor="black"), startangle=-40)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i] + ": " + str(np.round(data[i] * 100, 1)) + "%", xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, fontsize=12, **kw)

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    return imd



# 127 

def visualize_price(filtered_dataset):
    fs = filtered_dataset.copy()

    def wgs84_to_web_mercator(df, lon="longitude", lat="latitude"):
        k = 6378137
        df["x"] = df[lon] * (k * np.pi/180.0)
        df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
        return df

    CDMXhex = wgs84_to_web_mercator(fs)
    x = CDMXhex['x']
    y = CDMXhex['y']
    tile_provider = get_provider(Vendors.OSM)

    palette = RdBu[11]

    title = "Price of Airbnb Listings in "
    if fs["neighbourhood"].nunique() == 1:
        title = title + fs["neighbourhood"].iloc[0] + ", "
    for i in fs["neighbourhood_group"].unique():
        title = title + i + ", "
    title = title + "NYC"

    p = figure(title=title, match_aspect=False, tools = ["pan", "reset", "save", "wheel_zoom"], 
               x_axis_type="mercator", y_axis_type="mercator",plot_width=1050,plot_height=600)

    p.grid.visible = True

    r, bins = p.hexbin(x, y, size=127,
                       line_color="white", line_alpha=0.2,
                       palette=palette, hover_color="pink", alpha=0.7, hover_alpha=0.2)
    #print(bins["q"] + bins["r"])
    p.add_tools(HoverTool(
        tooltips=[("Price", "@g"), ("Region", "@neighbourhood")],
        show_arrow=True, mode="mouse", point_policy="follow_mouse", renderers=[r]))

    r = p.add_tile(tile_provider)
    r.level = "underlay"
    color_mapper = linear_cmap(
        field_name='price', palette=palette, low=fs['price'].min(), high=fs['price'].max())
    color_bar = ColorBar(color_mapper=color_mapper['transform'],
                         formatter=NumeralTickFormatter(format='0.0[0000]'),
                         label_standoff=13, width=8, location=(0, 0), title="price")
    p.add_layout(color_bar, 'right')
    output_notebook()
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    p.title.align = "right"
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.toolbar.logo = None
    script1, div1 = components(p)
    cdn_js = CDN.js_files[0]
    return script1, div1, cdn_js