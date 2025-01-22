import numpy as np
import scipy.ndimage

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool,LabelSet

from matipo.experiment.plot_interface import PlotInterface
from matipo.experiment.models import PLOT_COLORS, SITickFormatter, SIHoverFormatter

class LogLogMapPlot(PlotInterface):
    def __init__(self, figure_opts={}, image_opts={}, interpolate=4):
        self.interpolate = interpolate
        
        _figure_opts = dict(
            title='Map',
            x_axis_type='log',
            x_axis_label='x',
            y_axis_type='log',
            y_axis_label='y',
            sizing_mode='stretch_both',
            min_height=400,
            min_width=400,
            toolbar_location="above",
            tools='pan,wheel_zoom,box_zoom,reset,save'
        )
        _figure_opts.update(figure_opts) # allow options to be overriden
        
        self._image_opts = dict(
            palette='Greys256'
        )
        self._image_opts.update(image_opts) # allow options to be overriden
        
        self.fig = figure(**_figure_opts)
        self.fig.toolbar.logo = None
        self.fig.toolbar.active_drag = self.fig.tools[2]
        self.source = None

    def update(self, data, x0, x1, y0, y1):
        data = scipy.ndimage.zoom(data, self.interpolate, order=3)
        self.state = dict(
            d=[data],
            x=[x0],
            y=[y0],
            dw=[x1-x0],
            dh=[y1-y0]
        )
    
    @property
    def state(self):
        return self.source.data
    
    @state.setter
    def state(self, data):
        if self.source is None:
            self.source = ColumnDataSource(data=data)
            self.image = self.fig.image(image='d', x='x', y='y', dw='dw', dh='dh', source=self.source, **self._image_opts)
            self.image_hover = HoverTool(
                tooltips=[
                    (f'{self.fig.yaxis.axis_label}', '$y'),
                    (f'{self.fig.xaxis.axis_label}', '$x')
                ], 
                renderers=[self.image])
            self.fig.add_tools(self.image_hover)
        else:
            self.source.data = data

    def __call__(self):
        return self.fig