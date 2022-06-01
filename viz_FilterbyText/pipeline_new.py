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


from bokeh.palettes import RdBu


# def plot_bokeh_smalldf(dataframe):
#     fs = dataframe.copy()
#     df = fs
#     chosentile = get_provider(Vendors.OSM)
#     palette = RdBu[7]
#     source = ColumnDataSource(data=df)
#     # Define color mapper - which column will define the colour of the data points
#     color_mapper = linear_cmap(
#         field_name='price', palette=palette, low=df['price'].min(), high=df['price'].max())

#     # Set tooltips - these appear when we hover over a data point in our map, very nifty and very useful
#     tooltips = [("Price", "@price"), ("Region", "@neighbourhood"),
#                 ("Keywords", "@title_split"), ("Number of Reviews", "@number_of_reviews")]
#     # Create figure
#     p = figure(title='Airbnb Listings Selected',
#                x_axis_type="mercator",
#                y_axis_type="mercator",
#                x_axis_label='Longitude',
#                y_axis_label='Latitude',
#                tooltips=tooltips)
#     # Add map tile
#     p.add_tile(chosentile)
#     # Add points using mercator coordinates
#     sz = 2
#     if (len(fs) < 70) or (fs["neighbourhood"].nunique() == 1):
#         sz = 12
#     p.circle(x='mercator_x', y='mercator_y', color=color_mapper,
#              source=source, size=sz, fill_alpha=0.9)
#     # Defines color bar
#     color_bar = ColorBar(color_mapper=color_mapper['transform'],
#                          formatter=NumeralTickFormatter(format='0.0[0000]'),
#                          label_standoff=13, width=8, location=(0, 0), title="price")
#     p.add_layout(color_bar, 'right')
#     output_notebook()
#     p.xgrid.grid_line_color = None
#     p.ygrid.grid_line_color = None
#     p.axis.visible = False
#     p.title.align = "right"
#     show(p)


# def sort_keys(ls, df):
#     all_room_df = df[df.title_split.str.contains('|'.join(ls))]
#     return all_room_df


# def viz_key_df(ls, df):
#     key_df = sort_keys(ls, df)
#     plot_bokeh_smalldf(key_df)
#     return


# ## 可视化所选择的Airbnb
#
# function call: viz_key_df(room_list, fs)
#
# room_list: 关键词
#
# fs: 根据用户输入筛选过后的dataset


# room_list = ['bedroom', 'bedrooms', 'bed',
#              'beds', 'bdrs', 'bdr', 'room', 'rooms']
# fs = df_new.copy()
# fs = fs[fs["neighbourhood"].isin(["Kips Bay"])]

# viz_key_df(room_list, fs)


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

    p = figure(title=title, match_aspect=False,
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
    #output_notebook()
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    p.title.align = "right"

    script1, div1 = components(p)
    cdn_js = CDN.js_files[0]
    # cdn_css = CDN.css_files[0]
    return script1, div1, cdn_js

# # 可视化Number of Airbnbs in selected regions
#
# function call: visualize_count(fs)
#
# fs: 根据用户输入筛选后的dataset


# fs = df_new.copy()
# fs = fs[fs["neighbourhood_group"].isin(["Manhattan"])]
# fs = fs[fs["neighbourhood"].isin(["Kips Bay"])]

# visualize_count(fs)


# fs = df_new.copy()
# rt = fs.groupby("room_type").count()
# rt.reset_index(inplace=True)

# rt.iloc[:, 0:2]


def donut(dataset):
    fs = dataset.copy()
    # change figsize, must be two same numbers since it's square.
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))
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


# # 可视化 Room_type
#
# function call: donut(dataset)


# fs = df_new.copy()  # filtered dataset
# donut(fs)


# # # 之后的都不用管


# column_name = rt.columns[1]
# rt["angle"] = rt[column_name] / rt[column_name].sum() * 2 * np.pi
# rt["percentage"] = rt[column_name] / rt[column_name].sum() * 50 * 2
# rt['color'] = Category20c[len(rt)]

# p = figure(plot_height=350, title="", toolbar_location=None,
#            tools="hover", tooltips="@room_type: @percentage{0.f} %", x_range=(-.5, .5))

# p.annular_wedge(x=0, y=1,  inner_radius=0.15, outer_radius=0.25, direction="anticlock",
#                 start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
#                 line_color="white", fill_color='color', legend='room_type', source=rt)
# p.axis.axis_label = None
# p.axis.visible = False
# p.grid.grid_line_color = None
# show(p)


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

    p = figure(title=title, match_aspect=False,
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

    script1, div1 = components(p)
    cdn_js = CDN.js_files[0]
    # cdn_css = CDN.css_files[0]
    return script1, div1, cdn_js

# visualize_price(fs)


# def visualize_count(filtered_dataset):
#     fs = filtered_dataset

#     def wgs84_to_web_mercator(df, lon="longitude", lat="latitude"):
#         k = 6378137
#         df["x"] = df[lon] * (k * np.pi / 180.0)
#         df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
#         df["price"] = df[lon] * 0
#         return df

#     CDMXhex = wgs84_to_web_mercator(fs)
#     x = CDMXhex['x']
#     y = CDMXhex['y']
#     tile_provider = get_provider(Vendors.OSM)

#     palette = list(reversed(GnBu[9]))

#     title = "Number of Airbnb Listings in "
#     if fs["neighbourhood"].nunique() == 1:
#         title = title + fs["neighbourhood"].iloc[0] + ", "
#     if fs["neighbourhood_group"].nunique() < 5:
#         for i in fs["neighbourhood_group"].unique():
#             title = title + i + ", "
#     title = title + "NYC"

#     p = figure(title=title, match_aspect=False,
#                x_axis_type="mercator", y_axis_type="mercator")

#     p.grid.visible = True

#     sz = 1000
#     if fs["neighbourhood_group"].nunique() == 1:
#         sz = 500
#     if fs["neighbourhood"].nunique() == 1:
#         sz = 127
#     r, bins = p.hexbin(x, y, size=sz,
#                        line_color="white", line_alpha=0.2,
#                        palette=palette, hover_color="pink", alpha=0.7, hover_alpha=0.2)
#     p.add_tools(HoverTool(
#         tooltips=[("Count", "@c")],
#         show_arrow=True, mode="mouse", point_policy="follow_mouse", renderers=[r]))

#     r = p.add_tile(tile_provider)
#     r.level = "underlay"
#     color_mapper = linear_cmap(field_name='price', palette=palette,
#                                low=bins["counts"].min(), high=bins["counts"].max() + 1)
#     color_bar = ColorBar(color_mapper=color_mapper['transform'],
#                          formatter=NumeralTickFormatter(format='0.0[0000]'),
#                          label_standoff=13, width=8, location=(0, 0), title="count")
#     p.add_layout(color_bar, 'right')
#     output_notebook()
#     p.xgrid.grid_line_color = None
#     p.ygrid.grid_line_color = None
#     p.axis.visible = False
#     p.title.align = "right"
#     show(p)


# fs = df_new.copy()
# fs = fs[fs["neighbourhood_group"].isin(["Manhattan"])]
# fs = fs[fs["neighbourhood"].isin(["Kips Bay"])]

# visualize_count(fs)
