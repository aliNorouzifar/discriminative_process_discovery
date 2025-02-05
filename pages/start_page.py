from dash import html, dcc
import dash_bootstrap_components as dbc


layout = dbc.Container(
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
        # Introduction Section
        html.Div(
            className="intro-section",
            children=[
                html.H2("Welcome to IMr!", className="section-title"),
                html.P(
                    "IMr enhances process discovery by integrating domain knowledge and process rules, allowing for the creation of high-quality process models that align with both event data and expert knowledge.",
                    className="content",
                ),
                dcc.Link(
                    "Get Started",
                    href="/main",
                    className="cta-link"
                ),
            ],
        ),
        # Background and Context Section
        html.Div(
            className="context-section",
            children=[
                html.H2("Background and Context", className="section-title"),
                html.P(
                    "This tool is based on a novel framework introduced in a recent study that focuses on improving process discovery by leveraging rules alongside event logs. These rules, which define relationships between activities, can be discovered automatically or provided by domain experts, and IMr uses them to guide the process discovery workflow.",
                    className="content",
                ),
                html.Ul(
                    children=[
                        html.Li("Integrates discovered or user-defined rules into the process discovery workflow."),
                        html.Li(
                            "Employs a divide-and-conquer strategy, using rules to guide the selection of process structures."),
                        html.Li(
                            "Discovers high-quality imperative process models, such as BPMN models and Petri nets."),
                    ],
                    className="feature-list",
                ),
                html.P(
                    "The IMr framework has been evaluated on several real-world event logs, demonstrating that the discovered models better align with the provided rules without compromising their conformance to the event log.",
                    className="content",
                ),
                html.P(
                    "Furthermore, the evaluation shows that high-quality rules can improve model quality across well-known conformance metrics, highlighting the importance of integrating domain knowledge into process discovery.",
                    className="content",
                ),
                html.P(
                    "IMr is implemented as an open-source tool to enable broader applicability in real-world scenarios.",
                    className="content",
                ),
            ],
        ),
        # Example Section
        html.Div(
            className="example-section",
            children=[
                html.H2("Try the Tool with an Example", className="section-title"),
                html.P(
                    "Download the BPI challenge 2012 event log using the link below and try the tool.",
                    className="content",
                ),
                html.A(
                    "Download Sample File",
                    href="/assets/BPI_Challenge_2012_AO.xes",
                    download="BPI_Challenge_2012_AO.xes",
                    className="download-link",
                ),
            ],
        ),
    ],
)