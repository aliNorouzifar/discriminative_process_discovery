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
        # About Me Section
        html.Div(
            className="about-me-section",
            children=[
                html.H1("About Me", className="header"),
                html.P(
                    "Welcome! My name is Ali Norouzifar, and I am a researcher with a passion for advancing process mining techniques.",
                    className="content",
                ),
                html.P(
                    "Feel free to reach out with any questions or feedback about my work or the tool. Below are my contact details and professional profiles:",
                    className="content",
                ),
                # Links Section with Font Awesome Icons
                html.Div(
                    className="links-section",
                    children=[
                        html.A(
                            children=[
                                html.I(className="fas fa-envelope"),  # Email Icon
                                " Email Me",
                            ],
                            href="mailto:ali.norouzifar@pads.rwth-aachen.de",
                            className="link-item",
                            target="_blank",
                        ),
                        html.A(
                            children=[
                                html.I(className="fab fa-linkedin"),  # LinkedIn Icon
                                " LinkedIn",
                            ],
                            href="https://www.linkedin.com/in/ali-norouzifar/",
                            className="link-item",
                            target="_blank",
                        ),
                        html.A(
                            children=[
                                html.I(className="fab fa-github"),  # GitHub Icon
                                " GitHub",
                            ],
                            href="https://github.com/aliNorouzifar/",
                            className="link-item",
                            target="_blank",
                        ),
                        html.A(
                            children=[
                                html.I(className="fas fa-graduation-cap"),  # Google Scholar Icon
                                " Google Scholar",
                            ],
                            href="https://scholar.google.com/citations?user=veUzbLgAAAAJ&hl=en&oi=ao",
                            className="link-item",
                            target="_blank",
                        ),
                        html.A(
                            children=[
                                html.I(className="fas fa-link"),  # PADS Page Icon
                                " PADS Page",
                            ],
                            href="https://www.pads.rwth-aachen.de/cms/PADS/Der-Lehrstuhl/Team/Wissenschaftliche-Mitarbeiter/~nvxrd/Ali-Norouzifar/lidx/1/",
                            className="link-item",
                            target="_blank",
                        ),
                    ],
                ),
            ],
        ),
    ],
)

