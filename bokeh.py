from bokeh.io import output_file, show #output_notebook #for jupyter notebooks
from bokeh.plotting import figure

plot = figure(plot_width=400, tools='pan, box_zoom', x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)') #x_axis_type='datetime'

plot.circle(x=1,y=[8,6,5,2,3],size=[10,20,30,40,50],color='', fill_color='') #x and y coordinates of the circle
plot.line(x,y,line_width=3)
#markers
#asterisk, circle, circle_cross, circle_x, cross, diamond, diamond_cross, inverted_triangle, square, square_cross, square_x, triangle, x

output_file('circle.html')
show(plot)

"""patches"""
xs = [ [1,1,2,2], [2,2,4], [2,2,3,3] ]
ys = [ [2,5,5,2], [3,5,5], [2,3,4,2] ]

plot.patches(xs,ys, fill_color=['red','green','blue'], line_color='white')

#other glyphs
#annulus, annular_wedge, wedge, rect, quad, vbar, hbar, image, image_rgba, image_url, patch, patches, line, multi_line, circle, oval, ellipse, arc, quadratic, bezier


"""bokeh columndatasource"""
from bokeh.models import ColumnDataSource
source=ColumnDataSource(data={'x':[1,2,3,4,5],'y':[1,2,3,4,5]})
#or
source=ColumnDataSource(df)

#different code given
from bokeh.plotting import ColumnDataSource
source = ColumnDataSource(df)
p.circle(source=source, color='color', size=8, x='Year', y ='Time')

"""interactivity using tools"""
plot = figure(tools='box_select,lasso_select')
plot.circle(x, y, selection_color='red', nonselection_fill_alpha=0.2, nonselection_fill_color='gray') #x and y coordinates of the circle
from bokeh.models import HoverTool
hover = HoverTool(tooltips=None, mode='hline')
plot = figure(tools=[hover, 'crosshair'])
plot.circle(x,y,size=15,hover_color='red') #hover_...

#add tools
p.add_tools(hover)

"""Color mapping"""
from bokeh.models import CategoricalColorMapper
mapper = CategoricalColorMapper(factors=['setosa','virginica','versicolor'],palette=['red','green','blue'])
#Note: there are other color palettes that can be used as well..
plot.circle('columna','columnb',size=10,source=source,color={'field':'species','transform':mapper})

"""layouts"""
from bokeh.layouts import row, column #column
layout = row(column(p1,p2),p3)
#row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

from bokeh.layouts import gridplot
layout = gridplot([[None,p1],[p2,p3]], toolbar_location=None)
#tabbed layout
from bokeh.models.widgets import Tabs, Panel
first = Panel(child=row(p1,p2), title='first')
second = Panel(child=row(p3), title='second')
layout = Tabs(tabs=[first, second])
show(layout)

"""links panning and links brushing"""
#displayed ranges to stay synchronised
#or pan together

p3.x_range = p2.x_range = p1.x_range
p3.y_range = p2.y_range = p1.y_range

#for link brushing (same selection)
#create figures so they share the same source

#annotation and legends
plot.circle(......., legend='column')
plot.legend.location=top_left
p.legend.background_fill_color='lightgray'
"""hover tooltips"""
from bokeh.models import HoverTool
hover=HoverTool(tooltips=[('species name','@species')...]) #@ represents the column
plot = figure(tools=[hover,'pan','wheel_zoom'])

"""sliders"""
slider1 = Slider(title='slider1',start=0,end=10,step=0.1,value=2)

from bokeh.io import curdoc
#create pltos widgets
#add callbacks
#arrange plots/widgets in layouts
curdoc().add_root(layout)
#bokeh serve --show myapp.py
#bokeh serve --show myappdir/

def callback(attr, old, new):
	N = slider.value
	source.data = {'x':random(N), 'y':random(N)}
slider.on_change('value',callback) #whenever the slider's value changes, callback
layout = column(slider, plot)
curdoc().add_root(layout)

"""dropdowns"""
from bokeh.models import Select
menu = Select(options=['...',...] value='...', title='title')
def callback(attr,old,new):
	if menu.value == '...': f=random #you can change the options menu.options
	else: f=...
	source.data = {'x':f(size=N), 'y':f(size=N)}
menu.on_change('value', callback)
layout = column(menu, plot)
curdoc().add_root(layout)

"""buttons"""
from bokeh.models import Button
button= Button(label='Press me')
def update(): #no arguments
	#...
button.on_click(update)

#others
#CheckboxGroup, RadioGroup, Toggle
#Toggle(label='', button_type='success')
#CheckboxGroup(labels=['',...])
#RadioGroup(labels=[...])
#def callback(active):
#	...
