# Run this app with `python mockup.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import itertools
import random

from transformation import Transformation as Tr
from transformation import Rect
from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO)

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

    def add_constraint( self, blks, axis, alignment):
        self.children = blks
        self.axis = axis
        self.alignment = alignment
        self.transformations = None

    def add_child( self, blk, tr):
        self.children.append( blk)
        self.transformations.append( tr)

    def update_bbox(self):
        if self.children:
            for blk in self.children:
                blk.update_bbox()
                assert 0 == blk.llx
                assert 0 == blk.lly

            if self.axis is not None:
                assert self.alignment is not None
                M = None
                if self.axis == 'h':
                    for blk in self.children:
                        cand = blk.ury - blk.lly
                        if M is None or cand > M: M = cand
                    
                    x = 0
                    self.transformations = []
                    for blk in self.children:
                        if self.alignment == 'b':
                            self.transformations.append( Tr(oX=x))
                        elif self.alignment == 't':
                            self.transformations.append( Tr(oX=x,oY=M-blk.ury))
                        else:
                            assert False, self.alignment
                        x += blk.urx
                elif self.axis == 'v':
                    for blk in self.children:
                        cand = blk.urx - blk.llx
                        if M is None or cand > M: M = cand
                    
                    y = 0
                    self.transformations = []
                    for blk in self.children:
                        if self.alignment == 'l':
                            self.transformations.append( Tr(oY=y))
                        elif self.alignment == 'r':
                            self.transformations.append( Tr(oX=M-blk.urx,oY=y))
                        else:
                            assert False, self.alignment
                        y += blk.ury
                else:
                    assert False, self.axis

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

def block_v( nm, blks, alignment='l'):
    mid = Block(nm)
    mid.add_constraint( blks, 'v', alignment)
    return mid

def block_h( nm, blks, alignment='b'):
    mid = Block(nm)
    mid.add_constraint( blks, 'h', alignment)
    return mid

def polish( expr):
    """
    [ 'v', 'h', C, D, 'h', A, B]
"""
    stack = []
    for idx,x in reversed(list(enumerate(expr))):
        if x == 'v' or x == 'h':
            assert len(stack) >= 2
            e0 = stack.pop()
            e1 = stack.pop()
            if x == 'v':
                stack.append( block_v( f'w{idx}', [e1,e0]))
            elif x == 'h':
                stack.append( block_h( f'w{idx}', [e0,e1]))
            else:
                assert False, x
        else:
            stack.append( x)
    
    assert 1 == len(stack)
    return stack[0]

def all_possible_trees(n):
    if n == 1:
        yield 'l'
    for split in range(1, n):
        gen_left = all_possible_trees(split)
        gen_right = all_possible_trees(n-split)
        for left, right in itertools.product(gen_left, gen_right):
            yield 'o' + left + right

def all_possible_polish_strings(s):
    n = len(s)
    g0 = all_possible_trees(n)
    g1 = itertools.product( *([['h','v']]*(n-1)))
    g2 = itertools.permutations(s)
    
    for t, os, ls in itertools.product( g0, g1, g2):
        r, i1, i2 = '', 0, 0
        for c in t:
            if c == 'o':
                r += os[i1]
                i1 += 1
            elif c == 'l':
                r += ls[i2]
                i2 += 1

        assert i1 == len(os)
        assert i2 == len(ls)

        yield r

def test_A():
    lst = list(all_possible_polish_strings("ABCD"))
    s = set(lst)
    assert len(lst) == len(s)

def build_block():
    A = Block( "A", w=2, h=3)
    B = Block( "B", w=3, h=4)
    C = Block( "C", w=3, h=8)
    D = Block( "D", w=2, h=1)
    E = Block( "E", w=4, h=7)
    return A, B, C, D, E

def polish_str( s):
    tbl = { blk.nm : blk for blk in build_block()}
    return polish( [tbl.get(c, c) for c in s])

blocks = [ polish_str( 'vhCDhAB'),
           polish_str( 'hhvBACD'),
           polish_str( 'hvvDBAC'),
           polish_str( 'vvvDCBA'),
           polish_str( 'hAhBhCD')
]

blocks = [ polish_str(s) for s in all_possible_polish_strings("ABCDE")]

logging.info( f'Generated {len(blocks)} polish strings...')

for blk in blocks:
    blk.update_bbox()

logging.info( f'Resolving coordinates...')

placements = [ list(blk.gen_rects( Tr())) for blk in blocks]

logging.info( f'Generating placements (rectangle lists)...')

pairs = [ (random.gauss(blk.urx,.1),random.gauss(blk.ury,0.1)) for blk in blocks]

histo = defaultdict(list)
for blk in blocks:
    histo[(blk.urx,blk.ury)].append(blk)

logging.info( f'histo: {[(k, len(v)) for k,v in histo.items()]}')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.DataFrame( data=pairs, columns=['width','height'])

fig = px.scatter(df, x="width", y="height", width=600, height=600)

fig.update_traces( marker=dict(size=3))
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

    colors = {'A': 'Plum', 'B': 'Khaki', 'C': 'SpringGreen', 'D': 'Salmon', 'E': 'SteelBlue'}

    for named_rect in placements[idx]:
        nm, [x0, y0, x1, y1] = named_rect
        x = [x0, x1, x1, x0, x0]
        y = [y0, y0, y1, y1, y0]
        fig.add_trace( go.Scatter(x=x, y=y,
                                   mode='lines', fill='toself',
                                   fillcolor=colors[nm],
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
