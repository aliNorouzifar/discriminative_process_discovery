from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_uploader as du
import json
import pandas as pd
import dash_interactive_graphviz
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
                    html.H1("Rule-guided Process Discovery", className="tool-name"),
                    html.P(
                        "A cutting-edge tool for discovering process models from event logs considering rules as input.",
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
                    get_upload_component("upload-data"),
                ],
            ),
            # Parameters Section
            html.Div(
                className="parameters-wrapper",
                children=[
                    html.Hr(),

                    html.Div(id="output-data-upload2", className="parameter-block card"),
                    # html.Hr(),
                    html.Div(id="output-data-upload4", className="parameter-block card"),
                    # html.Hr(),
                    html.Div(id="output-data-upload6", className="parameter-block card"),
                    html.Div(id="output-data-upload8", className="parameter-block card"),
                    html.Div(id="output-data-upload10", className="parameter-block card"),
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
                    html.Div(id="petri_net1", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload5", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload7", className="visualization-block"),
                ],
            ),
        ],
    )

def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=500,
        chunk_size=100,
        max_files=1,
        filetypes=["xes"],
        upload_id="event_log",
    )

def rule_src_selection():
    return html.Div([
        html.H4("What is the rule source?:",className="parameter-name"),
        dcc.RadioItems(
            id='rule_src',
            options=[
                {'label': 'Minerful!', 'value': 'Minerful'},
                {'label': 'Local drive!', 'value': 'manual'},
                {'label': 'no rules!', 'value': 'no_rule'},
            ],
            value=["manual"],  # Default selected values
            labelStyle={'display': 'flex', 'alignItems': 'center', 'white-space': 'nowrap'},
            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}
        )
    ])


def show_rule_uploder():
    return html.Div(
                className="upload-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Upload the Rules", className="section-title"),
                    ),
                    du.Upload(
                        id="rule_upload",
                        max_file_size=500,
                        chunk_size=100,
                        max_files=1,
                        filetypes=["json"],
                        upload_id="rules",
                    )
                ],
            )


def show_Minerful_params():
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Minerful Parameters", className="section-title"),
                    ),
                    html.Hr(),
                    html.H4("support:", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='support_val',
                            type="number",
                            min=0,
                            max=1,
                            value=0.005,
                            size=200,
                            step=0.005,
                        ),
                        html.Div(id='numeric-input-output-1')
                    ]),
                    html.Hr(),
                    html.H4("confidence:", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='confidence_val',
                            type="number",
                            min=0,
                            max=1,
                            value=0.99,
                            size=200,
                            step=0.01,
                        ),
                        html.Div(id='numeric-input-output-2')
                    ]),
                    html.Hr(),
                    html.Button(id="run_Minerful", children="Run Minerful", className="btn-primary", n_clicks=0),
                    # html.Hr(),
                ]
            )
        ],
            className="flex-column align-center")
    ])



def IMr_params_show():
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("IMr Parameters", className="section-title"),
                    ),
                    html.Hr(),
                    html.H4("penalty dimension:", className="parameter-name"),
                    dcc.Dropdown(id='dimension', options=[{'label': x, 'value': x} for x in json.loads(redis_client.get('dimensions'))], value='support'),
                    html.Hr(),
                    html.H4("absence threshold:", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='absence_thr',
                            type="number",
                            min=-1,
                            max=1,
                            value=0.0,
                            step=0.05
                        ),
                    ]),
                    html.Hr(),
                    html.H4("IMr sup:", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='sup_IMr_val',
                            type="number",
                            min=0,
                            max=1,
                            value=0.2,
                            step= 0.1
                        ),
                    ]),
                    html.Hr(),
                    html.Button(id="run_IMr_selector", children="Run IMr", className="btn-primary", n_clicks=0),
                    # html.Hr(),
                ]
            )
        ],
            className="flex-column align-center")
    ])

def IMr_no_rules_params_show():
    return html.Div([
            html.Div([
                html.Div(
                    className="parameter-container",
                    children=[
                        html.Div(
                            className="section-header",
                            children=html.H4("IMr Parameters", className="section-title"),
                        ),
                        # html.Hr(),
                        html.H4("penalty dimension:", className="parameter-name", style={'display': 'none'}),
                        dcc.Dropdown(id='dimension', options=[{'label': x, 'value': x} for x in
                                                              json.loads(redis_client.get('dimensions'))],
                                     value='support', style={'display': 'none'}),
                        # html.Hr(),
                        html.H4("absence threshold:", className="parameter-name", style={'display': 'none'}),
                        html.Div([
                            dcc.Input(
                                id='absence_thr',
                                type="number",
                                min=-1,
                                max=1,
                                value=0.0,
                                step=0.05
                            , style={'display': 'none'}),
                        ]),
                        html.H4("IMr sup:", className="parameter-name"),
                        html.Div([
                            dcc.Input(
                                id='sup_IMr_val',
                                type="number",
                                min=0,
                                max=1,
                                value=0.2,
                                step= 0.1
                            ),
                        ]),
                        html.Hr(),
                        html.Button(id="run_IMr_selector", children="Run IMr", className="btn-primary", n_clicks=0),
                        # html.Hr(),
                    ]
                )
            ],
                className="flex-column align-center")
        ])

def rule_related_statistics_show(N_rules, N_dev, support_cost,confidence_cost, dev_list):
    statistics=[
        f"The Number of rules used in the discovery is: {N_rules}",
        f"The Number of deviating rules considering the discovered model is: {N_dev}",
        f"The Number of deviation ratio considering the discovered model and support is: {support_cost}",
        f"The Number of deviation ratio considering the discovered model and confidence is: {confidence_cost}",
        f"The most significant deviations (based on support cost), maximum 10 of them:"
    ]
    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Rule-based Statistics", className="section-title"),
            ),
            html.Ul([html.Li(line, className="list-item") for line in statistics]),
            html.Ul([html.Li(line, className="list-item") for line in dev_list]),
            html.Button(id="ccf_show", children="Show conformance checking results",
                        className="btn-secondary", n_clicks=0)
        ]
    )


def conformance_related_statistics_show(fit, prc):
    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Conformance Checking Results", className="section-title"),
            ),
            html.Ul(html.Li(f"The fitness value is: {fit}", className="list-item")),
            html.Ul(html.Li(f"The precision value is: {prc}", className="list-item"))
        ]
    )

def show_petri_net(gviz):
    return html.Div(
                        className="visualization-wrapper",
                        children=[
                            # Graph and slider container
                            html.Div(
                                className="graph-slider-container",
                                style={"display": "flex", "alignItems": "center"},
                                children=[
                                    # Zoom slider on the left
                                    html.Div(
                                        className="slider-container",
                                        style={"width": "10%", "marginRight": "10px"},
                                        children=[
                                            dcc.Slider(
                                                id="zoom-slider",
                                                min=1.0,  # Minimum zoom level
                                                max=3.0,  # Maximum zoom level
                                                step=0.1,  # Increment steps
                                                value=1.0,  # Default zoom level
                                                marks={i: f"{i:.1f}" for i in [1.0, 1.5, 2.0,2.5,3.0]},
                                                vertical=True,  # Make it vertical
                                            ),
                                        ],
                                    ),
                                    # Graph visualization on the right
                                    html.Div(
                                        className="graph-container",
                                        style={
                                            "flexGrow": 1,
                                            "border": "1px solid #ddd",
                                            "height": "500px",
                                            "position": "relative",
                                            "overflow": "hidden",
                                        },
                                        children=[
                                            dash_interactive_graphviz.DashInteractiveGraphviz(
                                                id="gv",
                                                style={"transform": "scale(1)", "transformOrigin": "0 0"},
                                                dot_source=str(gviz),
                                                 engine = "dot"
                                            ),
                                        ],
                                    ),
                                ],
                            ),html.Button(id="stats_show", children="Show statistics",
                                className="btn-secondary", n_clicks=0),])