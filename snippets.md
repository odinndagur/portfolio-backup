# Code snippets

## Read data from GeoTIF into NP array
```python
from osgeo import gdal, gdal_array
from math import sqrt
import pandas as pd
import os
import numpy as np

filepath = '~/datafolder/IslandsDEMv1.0_2x2m_zmasl_isn2016_57.tif'

rasterArray = gdal_array.LoadFile(filepath)
raster = gdal.Open(filepath)
band = raster.GetRasterBand(1)

print(gdal.GetDataTypeName(band.DataType))
# Get nodata value from the GDAL band object
nodata = band.GetNoDataValue()

#Create a masked array for making calculations without nodata values
rasterArrayMasked = np.ma.masked_equal(rasterArray, nodata)
rasterArray[rasterArray < -100.0] = -100.0 # arbitrary value for the lowest values to avoid nodata
```

## Save np array to binary file (as single dimensional array)
```python
# Choose a subset of the array with a certain step size and save to custom binary file
meshdata = rasterArray[xmin:xmax:xstep,ymin:ymax:ystep]
with open('meshdata/' + name + '.dat', 'wb') as f:
    f.write(meshdata.tobytes())
```


```python
def generate_terrain_mesh(row,col):
    w = 251
    h = 251
    scale = 1
    plane_name = 'x{x}y{y}_plane_width{w}_height{h}_scale{scale}'.format(x=col,y=row,w=w,h=h,scale=scale)
    xmin = col * (w-1)
    ymin = row * (h-1)
    edges = []
    faces = []
    
    lastRowCol = 9
    if(row == lastRowCol):
        xmin -= 1
    if(col == lastRowCol):
        ymin -= 1
    vertices = [((x-xmin)*scale,(y-ymin)*scale,heights[x][y]*scale) for x in range(xmin,xmin+w) for y in range(ymin,ymin+h)]

    # make triangle faces
    vertexIndex = 0
    sz = int(sqrt(len(vertices)))
    for y in range(sz):
        for x in range(sz):
            if (x < sz - 1) and (y < sz - 1):
                # faces.append((vertexIndex,vertexIndex+width+1,vertexIndex+width)) #unity mode
                # faces.append((vertexIndex + width + 1, vertexIndex, vertexIndex+1)) #unity mode
                faces.append((vertexIndex+sz,vertexIndex+sz+1,vertexIndex)) #blender mode
                faces.append((vertexIndex+1, vertexIndex,vertexIndex + sz + 1)) #blender mode
            vertexIndex+=1

    new_mesh = bpy.data.meshes.new(plane_name)
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    # make object from mesh
    new_object = bpy.data.objects.new(plane_name, new_mesh)
    # make collection
    IslandsDEM_collection = bpy.data.collections.get('IslandsDEM')
    if not IslandsDEM_collection:
        IslandsDEM_collection = bpy.data.collections.new('IslandsDEM')
        bpy.context.scene.collection.children.link(IslandsDEM_collection)
    IslandsDEM_collection.objects.link(new_object)
    new_object.select_set(True)
    new_object.location = (col*w*scale - scale*col,row*h*scale - scale*row,0)
```