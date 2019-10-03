"""two plots"""
plt.axes([0.05,0.5,0.425,0.9]) #lower corners, width and height
#figure dimensions 0-1 used
plt.plot(t,temperature,'r')
plt.xlabel('')
plt.title('')
plt.axes([0.05,0.5,0.425,0.9])
plt.plot(t,temperature,'r')
plt.xlabel('')
plt.title('')
plt.show()

plt.subplot(2,1,1)
#...
plt.subplot(2,1,1)
#...
plt.tight_layout()