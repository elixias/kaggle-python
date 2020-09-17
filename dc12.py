"""stlyes"""
plt.style.available
plt.style.use('ggplot') #fivethirtyeight

"""colormaps"""
'jet', 'coolwarm', 'magma' and 'viridis'
'Greens', 'Blues', 'Reds', and 'Purples'
'summer', 'autumn', 'winter' and 'spring'

"""two plots"""
plt.axes([0.05,0.5,0.425,0.9]) #lower corners, width and height
#figure dimensions 0-1 used
plt.plot(t,temperature,'r',label='Temperature') #labelling for legends
plt.xlabel('')
plt.title('')
plt.axes([0.05,0.5,0.425,0.9])
plt.plot(t,temperature,'r')
plt.xlabel('')
plt.title('')
plt.show()

plt.subplot(2,1,1) #number of rows, cols and subplot # to activate
#...
plt.subplot(2,1,2)
#...
plt.tight_layout()

"""specifying limits of graph to display"""
plt.axis([xmin,xmax,ymin,ymax]) #axis('off')/equal, square, tight
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])

plt.savefig()

"""scatter"""
plt.scatter(x,y,marker='o', color='',label='')
...
plt.legend(loc='upper right')

"""annotate"""
plt.annotate('tetx', xy=(5.0,3.5), xytext=(4.25,4.0), arrowprops={'color':'red'}) #inputs are exact variable values, not coordinate values
#xy=point to annotate, xytext=position to place the text
"""linspace and meshgrid"""
u=np.linspace(-2,2,3)
v=np.linspace(-1,1,5)
X,Y=np.meshgrid(u,v)
Z=X**2/25+Y**2/4 #Z = np.sin(3*np.sqrt(X**2 + Y**2)) 
print('Z:\n', Z)
plt.set_cmap('grayscale')
plt.pcolor(X,Y,Z) #or or plt.pcolor(Z) plt.pcolor(Z,cmap='Blues') plt.imshow()
plt.colorbar()
plt.contour() # instead of pcolor, contour(Z,30)
#plt.contourf()
plt.axis('tight')
plt.show()

plt.hist2d(x,y,bins=(10,20))
plt.hist2d(hp,mpg,bins=(20,20),range=((40,235),(8,48)))
plt.hexbin(x,y,gridsize=(15,10))
plt.hexbin(hp,mpg,gridsize=(15,12),extent=(40,235,8,48))

img = plt.imread('x.jpg') #reads images
plt.imshow(img)
plt.axis('off') #hides axis when displaying image
#converting to grayscale
plt.set_cmap('grayscale')
collapsed = img.mean(axis=2)
uneven = collapsed[::4,::2]
plt.imshow(uneven,aspect=2.0) #or use extent=(0,680,0,480)
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

plt.twinx() #to overlay plots with different vertical scales on the same horizontal axis
