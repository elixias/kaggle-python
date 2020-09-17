data = [dict(
        type='choropleth',
        locations= mpi_region_amount.index,
        locationmode='country names',
        z=mpi_region_amount.values,
        text=mpi_region_amount.index,
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Top mpi_regions with amount(Mean value)'),
)]
layout = dict(title = 'Top mpi_regions with amount(Dollar value of loans funded in particular LocationName)',
             geo = dict(
            showframe = False,
            #showcoastlines = False,
            projection = dict(
                type = 'Mercatorodes'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)