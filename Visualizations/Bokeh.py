import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Select, CustomJS
from bokeh.layouts import column

# Global variables to store axis limits across multiple plots
global_x_min, global_x_max = None, None
global_y_min, global_y_max = None, None

def create_bokeh_plot(k_means, embeddings, df, current_id, doc_type, model_name):
    """
    Generates an interactive Bokeh plot of text embeddings with clustering.
    Ensures all plots share the same axis range for consistency.
    """

    global global_x_min, global_x_max, global_y_min, global_y_max

    # Convert embeddings to NumPy array
    embeddings = np.array(embeddings)

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=min(30, (len(embeddings) - 1) // 3), random_state=42)
    data_map = tsne.fit_transform(embeddings)

    # Perform K-Means clustering
    cluster_labels = k_means['labels_layers']

    # Store axis limits globally
    x_min, x_max = data_map[:, 0].min(), data_map[:, 0].max()
    y_min, y_max = data_map[:, 1].min(), data_map[:, 1].max()

    # Update global axis limits if this is the first plot or new extremes are found
    if global_x_min is None or x_min < global_x_min:
        global_x_min = x_min
    if global_x_max is None or x_max > global_x_max:
        global_x_max = x_max
    if global_y_min is None or y_min < global_y_min:
        global_y_min = y_min
    if global_y_max is None or y_max > global_y_max:
        global_y_max = y_max

    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame(data_map, columns=["x", "y"])
    df_plot["Title"] = df["Title"].values
    df_plot["Cluster"] = cluster_labels
    df_plot["Source"] = df["Source"].values  # Ensure DataFrame contains "Source" column

    # Convert to Bokeh DataSource
    source_all = ColumnDataSource(df_plot)
    source_reg = ColumnDataSource(df_plot[df_plot["Source"] == "Reg Media"])
    source_pro = ColumnDataSource(df_plot[df_plot["Source"] == "Pro Media"])
    source_sci = ColumnDataSource(df_plot[df_plot["Source"] == "Sci Media"])
    source_merged = ColumnDataSource(df_plot[df_plot["Source"] == "Merged"])

    # Create Bokeh figure with fixed axis ranges
    p = figure(title=f"{model_name} Embeddings",
               tools="pan,wheel_zoom,reset,save",
               width=800, height=600,
               tooltips=[("Title", "@Title"), ("Cluster", "@Cluster"), ("Source", "@Source")],
               x_range=(global_x_min, global_x_max),  # Keep x-axis range fixed
               y_range=(global_y_min, global_y_max))  # Keep y-axis range fixed

    scatter = p.scatter("x", "y", source=source_all, size=8, color="navy", alpha=0.6)

    # Dropdown selector for different sources
    select = Select(title="Select Data Source", value="All",
                    options=["All", "Reg Media", "Pro Media", "Sci Media", "Merged"])

    # JavaScript callback to update source dynamically
    callback = CustomJS(args=dict(source_all=source_all, source_reg=source_reg,
                                  source_pro=source_pro, source_sci=source_sci,
                                  source_merged=source_merged, scatter=scatter), code="""
        var data_source = cb_obj.value;
        if (data_source == "All") { scatter.data_source.data = source_all.data; }
        else if (data_source == "Reg Media") { scatter.data_source.data = source_reg.data; }
        else if (data_source == "Pro Media") { scatter.data_source.data = source_pro.data; }
        else if (data_source == "Sci Media") { scatter.data_source.data = source_sci.data; }
        else if (data_source == "Merged") { scatter.data_source.data = source_merged.data; }
        scatter.data_source.change.emit();
    """)

    select.js_on_change('value', callback)

    # Save and show plot
    output_file(f"visualizations/outputs/{current_id}_{doc_type}_{model_name}_bokehplot.html")
    show(column(select, p))

    print("Bokeh plot saved successfully.")
