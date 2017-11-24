import functions as f
import layers as l

def hello(x):
    print(x)


#f.Conv2D([1,2,3,4,5,6,7],3,1,0,1,0)
#f.Conv2D([1,2,3,4,5,6,7],3,2,0,1,0)
layer=l.ConvLayer(1,2,3,4,5,6)
print(layer.fieldSize)
