# Run this app with `python mockup.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from transformation import Transformation as Tr
from transformation import Rect

class Block:
    def __init__(self, nm, *, w=None, h=None):
        self.nm = nm
        self.llx, self.lly, self.urx, self.ury = None, None, None, None
        if w is not None:
            assert h is not None
            self.llx, self.lly, self.urx, self.ury = 0, 0, w, h
        self.children = []
        self.transformations = []
        self.axis = None
        self.alignment = None

    def add_child( self, blk, tr):
        self.children.append( blk)
        self.transformations.append( tr)

    def update_bbox(self):
        if self.children:
            self.llx, self.lly, self.urx, self.ury = None, None, None, None
            for (blk, local_tr) in zip(self.children,self.transformations):
                blk.update_bbox()
                llx, lly, urx, ury = tuple(local_tr.hitRect( Rect(ll=(blk.llx,blk.lly), ur=(blk.urx,blk.ury))).canonical().toList())
                if self.llx is None or llx < self.llx: self.llx = llx
                if self.lly is None or lly < self.lly: self.lly = lly
                if self.urx is None or urx > self.urx: self.urx = urx
                if self.ury is None or ury > self.ury: self.ury = ury


    def gen_rects( self, global_tr):
        if not self.children:
            yield self.nm, global_tr.hitRect( Rect(ll=(self.llx,self.lly), ur=(self.urx,self.ury))).canonical().toList()
        for (blk, local_tr) in zip(self.children,self.transformations):
            tr = global_tr.postMult(local_tr)
            yield from blk.gen_rects( tr)

def build_block0():
    top = Block("top")

    """(C|D)-(A|B)"""
    top.add_child( Block( "A", w=2, h=3), Tr(oX=0,oY=0))
    top.add_child( Block( "B", w=3, h=4), Tr(oX=2,oY=0))
    top.add_child( Block( "C", w=3, h=8), Tr(oX=0,oY=4))
    top.add_child( Block( "D", w=2, h=1), Tr(oX=3,oY=4))

    top.update_bbox()
    return top

def build_block1():
    top = Block("top")

    """(B-A)|C|D"""
    top.add_child( Block( "A", w=2, h=3), Tr(oX=0,oY=0))
    top.add_child( Block( "B", w=3, h=4), Tr(oX=0,oY=3))
    top.add_child( Block( "C", w=3, h=8), Tr(oX=3,oY=0))
    top.add_child( Block( "D", w=2, h=1), Tr(oX=6,oY=0))

    top.update_bbox()
    return top

def build_block2():
    top = Block("top")

    """(D-B-A)|C"""
    top.add_child( Block( "A", w=2, h=3), Tr(oX=0,oY=0))
    top.add_child( Block( "B", w=3, h=4), Tr(oX=0,oY=3))
    top.add_child( Block( "C", w=3, h=8), Tr(oX=3,oY=0))
    top.add_child( Block( "D", w=2, h=1), Tr(oX=0,oY=7))

    top.update_bbox()
    return top

def build_block3():
    top = Block("top")

    """D-C-B-A"""
    top.add_child( Block( "A", w=2, h=3), Tr(oX=0,oY=0))
    top.add_child( Block( "B", w=3, h=4), Tr(oX=0,oY=3))
    top.add_child( Block( "C", w=3, h=8), Tr(oX=0,oY=7))
    top.add_child( Block( "D", w=2, h=1), Tr(oX=0,oY=15))

    top.update_bbox()
    return top

def build_block4():
    top = Block("top")

    """A|B|C|D"""    
    top.add_child( Block( "A", w=2, h=3), Tr(oX=0,oY=0))
    top.add_child( Block( "B", w=3, h=4), Tr(oX=2,oY=0))
    top.add_child( Block( "C", w=3, h=8), Tr(oX=5,oY=0))
    top.add_child( Block( "D", w=2, h=1), Tr(oX=8,oY=0))

    top.update_bbox()
    return top

def test_Block():
    print( build_block0())

blocks = [build_block0(), build_block1(), build_block2(), build_block3(), build_block4()]

placements = [ list(blk.gen_rects( Tr())) for blk in blocks]

pairs = [ [blk.urx,blk.ury] for blk in blocks]


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.DataFrame( data=pairs, columns=['width','height'])

fig = px.scatter(df, x="width", y="height", width=600, height=600)

fig.update_traces( marker=dict(size=12))
fig.update_xaxes( rangemode="tozero")
fig.update_yaxes( rangemode="tozero")

app.layout = html.Div([
    html.Div([
        html.H2(children='Pareto Frontier'),
        dcc.Graph(
            id='width-vs-height',
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
    Input('width-vs-height', 'hoverData'))
def display_hover_data(hoverData):
    fig = go.Figure()

    if hoverData is None:
        return fig

    points = hoverData['points']
    assert 1 == len(points)
    idx = points[0]['pointNumber']

    for named_rect in placements[idx]:
        nm, [x0, y0, x1, y1] = named_rect
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
