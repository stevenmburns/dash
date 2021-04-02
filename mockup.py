# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def trans( named_r, o):
    nm, (p0,p1) = named_r
    return nm, (tuple( pp+oo for pp,oo in zip(p0,o)),tuple( pp+oo for pp,oo in zip(p1,o)))

A = ("A", ((0,0),(2,3)))
B = ("B", ((0,0),(3,4)))
C = ("C", ((0,0),(3,8)))
D = ("D", ((0,0),(2,1)))

# Prefix notation

placement0 = [ trans(A, (0,0)), 
               trans(B, (2,0)), 
               trans(C, (0,4)), 
               trans(D, (3,4))]

placement1 = [ trans(A, (0,0)), 
               trans(B, (0,3)), 
               trans(C, (3,0)), 
               trans(D, (6,0))]

placements = [placement0,placement1,placement0,placement1]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.DataFrame( data=[ [10,1],[5,2],[2,5],[1,10]], columns=['area','hpwl'])

fig = px.scatter(df, x="area", y="hpwl", width=600, height=600)

fig.update_traces( marker=dict(size=12))
fig.update_xaxes( rangemode="tozero")
fig.update_yaxes( rangemode="tozero")

app.layout = html.Div([
    html.Div([
        html.H2(children='Pareto Frontier'),
        dcc.Graph(
            id='area-vs-hpwl',
            figure=fig
        )
    ], style={'display': 'inline-block'}),
    html.Div([    
        html.H2(children='Placement'),
        dcc.Graph(
            id='Placement'
        )
    ], style={'display': 'inline-block'})
])

@app.callback(
    Output('Placement', 'figure'),
    Input('area-vs-hpwl', 'hoverData'))
def display_hover_data(hoverData):
    fig = go.Figure()

    if hoverData is None:
        return fig

    points = hoverData['points']
    assert 1 == len(points)
    idx = points[0]['pointNumber']

    for named_rect in placements[idx]:
        nm, ((x0, y0), (x1, y1)) = named_rect
        x = [x0, x1, x1, x0, x0]
        y = [y0, y0, y1, y1, y0]
        fig.add_trace( go.Scatter(x=x, y=y,
                                   mode='lines', fill='toself',
                                   showlegend=False,
                                   name=f'{nm}'))

    fig.update_yaxes(
        scaleanchor='x',
        scaleratio = 1
    )

    fig.update_layout(
        autosize=False,
        width=600,
        height=600
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
