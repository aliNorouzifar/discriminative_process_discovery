from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash_daq as daq
import json
import pandas as pd
from prolysis.util.redis_connection import redis_client

def load_variables():
    try:
        with open("output_files/internal_variables.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return "No data file found."
    df = pd.read_json(data["df"], orient="split")
    data["df"] = df
    return data

def create_layout():
    return dbc.Container(
        className="page-container",
        children=[
            # Navigation Links
            html.Div(
                className="nav-links",
                children=[
                    dcc.Link("Introduction", href="/", className="nav-link"),
                    dcc.Link("XPVI", href="/XPVI", className="nav-link"),
                    dcc.Link("IMr", href="/IMr", className="nav-link"),
                    dcc.Link("About Me", href="/about_me", className="nav-link"),
                ],
            ),
            # Tool Name and Description
            html.Div(
                className="tool-name-container",
                children=[
                    html.H1("Process Variant Identification", className="tool-name"),
                    html.P(
                        "A cutting-edge tool for detecting and understanding process variability across performance dimensions.",
                        className="tool-subtitle",
                    ),
                ],
            ),
            # Main Content Area
            html.Div(
                className="flex-container",
                children=[
                    # Left Panel: Parameter Settings
                    create_left_panel(),
                    # Right Panel: Visualization Blocks
                    create_right_panel(),
                ],
            ),
        ],
    )


def create_left_panel():
    return html.Div(
        id="left-panel",
        className="left-panel-container",
        children=[
            # Header for Parameters Settings
            html.Div(
                className="panel-header",
                children=html.H4("Parameters Settings", className="panel-title"),
            ),
            # Upload Section
            html.Div(
                className="upload-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Upload the Event Log", className="section-title"),
                    ),
                    get_upload_component("event_log_upload"),
                ],
            ),
            # Parameters Section
            html.Div(
                className="parameters-wrapper",
                children=[
                    html.Hr(),
                    html.Div(id="output-data-upload102", className="parameter-block card"),
                    html.Div(id="output-data-upload104", className="parameter-block card"),
                    html.Div(id="output-data-upload106", className="parameter-block card"),
                    html.Div(id="output-data-upload108", className="parameter-block card"),
                    html.Div(id="output-data-upload110", className="parameter-block card"),
                    html.Button(id="latest_log", children="Show the Latest Log", className="btn-primary", n_clicks=0)
                ],
            ),
        ],
    )


def create_right_panel():
    return html.Div(
        id="right-panel",
        className="right-panel-container",
        children=[
            html.Div(
                className="panel-header",
                children=html.H4("Visualizations and Reports", className="panel-title"),
            ),
            html.Div(
                className="visualization-wrapper",
                children=[
                    html.Div(id="output-data-upload103", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload105", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload107", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload109", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload111", className="visualization-block"),
                    html.Hr(),
                    html.Div([
                        html.Pre(id="log-display",
                                 style={"whiteSpace": "pre-wrap", "height": "400px", "overflowY": "scroll"})
                    ])
                ],
            ),
        ],
    )

def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=800,
        chunk_size=100,
        max_files=1,
        filetypes=["xes"],
        upload_id="event_log",
    )

def parameters_view_PVI(max_par, columns):
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Process Variant Identification Parameters", className="section-title"),
                    ),
                    # html.Hr(),
                    html.H4("process indicator:", className="parameter-name"),
                    dcc.Dropdown(id='kpi', options=[{'label': x, 'value': x} for x in columns]),
                    html.Hr(),
                    html.H4("N. buckets:", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='n_bins',
                            type='number',
                            min=2,
                            max=max_par,
                            value=min(100, max_par),
                        ),
                        # html.Div(id='numeric-input-output-1')
                    ],  className="daq-numeric-input-container"),
                    html.Hr(),
                    html.H4("Window size:", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='w',
                            type = 'number',
                            min=0,
                            max=max_par / 2,
                            value=2,
                    ),
                        html.Div(id='numeric-input-output-2')
                    ]),
                    # html.Hr(),
                    # html.Button(id="run_PVI", children="Run PVI", className="btn-primary", n_clicks=0),
                ]
            )
        ],
        className="flex-column align-center")
    ])

def parameters_view_segmentation(max_dis):
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Segmentation parameter", className="section-title"),
                    ),
                    html.H4("significant distance:", className="parameter-name"),
                    html.Br(),
                    html.Div([
                        daq.Slider(
                            id='sig_dist',
                            min=0,
                            max=max_dis,
                            value=0.5* max_dis,
                            handleLabel={"showCurrentValue": True,"label": "VALUE"},
                            color="#007bff",
                            step=0.01
                        ),
                        html.Div(id='slider-output-container3')
                    ]),
                    html.Hr(),
                ]
            )
        ],
        className="flex-column align-center")
    ])

def parameters_feature_extraction():
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Feature Extraction Parameters", className="section-title"),
                    ),
                    # html.Hr(),
                    html.H4("theta_cvg for pruning?", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='theta_cvg',
                            type='number',
                            min=0,
                            max=1,
                            value=0.02,
                            step=0.01  # Specify the step size here
                        ),
                        html.Div(id='numeric-input-output-3')
                    ]),
                    html.Hr(),
                    html.H4("Number of Clusters?", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='n_clusters',
                            type='number',
                            min=0,
                            max=20,
                            value=5
                        ),
                        html.Div(id='numeric-input-output-4')
                    ]),
                    # html.Hr(),
                    # html.Button(id="minerful_run", children="Feature Extraction", className="btn-primary", n_clicks=0)
                ]
            )
        ],
        className="flex-column align-center")
    ])

def parameters_view_explainability():
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Explainability Extraction parameters", className="section-title"),
                    ),
                    # html.Hr(),
                    html.Hr(),
                    html.H4("Number of Clusters?", className="parameter-name"),
                    html.Div([
                        daq.NumericInput(
                            id='n_clusters',
                            min=0,
                            max=20,
                            value=5
                        ),
                        html.Div(id='numeric-input-output-4')
                    ]),
                    html.Hr(),
                    html.Button(id="XPVI_run", children="XPVI Run", className="btn-primary", n_clicks=0)
                ]
            )
        ],
        className="flex-column align-center")
    ])

def decl2NL_parameters():
    # data = load_variables()
    #
    # segments_count = data["segments_count"]
    # clusters_count = data["clusters_count"]

    segments_count = int(redis_client.get("segments_count"))
    clusters_count = int(redis_client.get("clusters_count"))
    return html.Div([
        html.Div(className="parameter-container",
            children=[
                html.Div(
                    className="section-header",
                    children=html.H4("Report Generation Parameters", className="section-title"),
                ),
                html.H4("Which segment?", className="parameter-name"),
                dcc.Dropdown(id='segment_number', options=[{'label': x, 'value': x} for x in range(1, segments_count + 1)]),
                html.Hr(),
                html.H4("Which cluster?", className="parameter-name"),
                dcc.Dropdown(id='cluster_number', options=[{'label': x, 'value': x} for x in range(1, clusters_count + 1)]),
                html.Hr(),
                html.Button(id="decl2NL_pars", children="Show decl2NL parameters!", className="btn-primary", n_clicks=0)
            ]
        )
    ])

''' Matplotlib Figures'''
# def PVI_figures(fig_src1, fig_src2):
#     return html.Div(
#         id="bottom-section",
#         className="page-container",
#         children=[
#             html.Div(
#                 className="section-header",
#                 children=html.H4("Process Variant Identification Visualizations", className="section-title"),
#             ),
#             # html.Img(id="bar-graph-matplotlib", src=fig_src1, className="figure figure-large"),
#             # html.Img(id="bar-graph-matplotlib2", src=fig_src2, className="figure figure-small"),
#             dcc.Graph(id='heatmap1', figure=fig_src1, className="figure figure-large"),
#             dcc.Graph(id='heatmap2', figure=fig_src2, className="figure figure-large"),
#             html.Button(id="X_parameters", children="Start The Explainability Extraction Framework!", className="btn-secondary", n_clicks=0)
#         ]
#     )

def PVI_figures_EMD(fig_src1, ):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("EMD-based Change Detection Visualizations", className="section-title"),
            ),
            dcc.Graph(id='heatmap1', figure=fig_src1, className="figure figure-large"),
            html.Button(id="Seg_parameters", children="Start Segmentation!", className="btn-secondary", n_clicks=0)
        ]
    )

def PVI_figures_Segments(fig_src2,peaks):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Segmentation Visualizations", className="section-title"),
            ),
            dcc.Graph(id='heatmap2', figure=fig_src2, className="figure figure-large"),
            html.Ul([html.Li(line, className="list-item") for line in peaks]),
            html.Button(id="export", children="Export the Segments as event logs.",
                        className="btn-secondary", n_clicks=0),
            html.Hr(),
            html.Button(id="X_parameters", children="Start The Explainability Extraction Framework!", className="btn-secondary", n_clicks=0)
        ]
    )


def XPVI_figures(fig_src3, fig_src4):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Explainability Extraction Visualizations", className="section-title"),
            ),
            # html.Img(id="bar-graph-matplotlib3", src=fig_src3, className="figure figure-large"),
            # html.Img(id="bar-graph-matplotlib4", src=fig_src4, className="figure figure-medium"),
            dcc.Graph(id='heatmap3', figure=fig_src3, className="figure figure-large"),
            dcc.Graph(id='heatmap4', figure=fig_src4, className="figure figure-large"),
            html.Button(id="decl2NL_framework", children="Convert Declare to Natural Language!", className="btn-secondary", n_clicks=0)
        ]
    )




def statistics_print(list_sorted, list_sorted_reverse):
    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Natual Language Report", className="section-title"),
            ),
            html.H4("Lowest Scores:", className='text-left bg-light mb-4'),
            html.Ul([html.Li(sentence, className="list-item") for sentence in list_sorted]),
            html.H4("Highest Scores:", className='text-left bg-light mb-4'),
            html.Ul([html.Li(sentence, className="list-item") for sentence in list_sorted_reverse])
        ]
    )
