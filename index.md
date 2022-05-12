# Compute Shader terrain adventures

![](D2FA980D-589B-492E-BDE6-01B5E3C24FFA.gif)

## python

```python
from osgeo import gdal, gdal_array
from math import sqrt
import pandas as pd
import os
import numpy as np


filepath = '/Users/odinndagur/Blender/2022/islandsdem-modular-heightmap/data/temp/IslandsDEMv1.0_2x2m_zmasl_isn2016_57_x_3_y_5.tif'
rasterArray = gdal_array.LoadFile(filepath)
raster = gdal.Open(filepath)
band = raster.GetRasterBand(1)

print(gdal.GetDataTypeName(band.DataType))
# Get nodata value from the GDAL band object
nodata = band.GetNoDataValue()

#Create a masked array for making calculations without nodata values
rasterArray = np.ma.masked_equal(rasterArray, nodata)
type(rasterArray)

# Check again array
min1 = rasterArray.min()
max1 = rasterArray.max()
print(min1,max1)


df = pd.DataFrame(rasterArray)
heights = df.iloc[:,1:].values


with open('binary-heights-file.dat','wb') as f:
    f.write(rasterArray[xstart:xend:xstep,ystart:yend:ystep].tobytes())
```

## blender mesh gen with python
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
    
    if(row == 9):
        xmin -= 1
    if(col == 9):
        ymin -= 1
    vertices = [((x-xmin)*scale,(y-ymin)*scale,heights[x][y]*scale) for x in range(xmin,xmin+w) for y in range(ymin,ymin+h)]
    #print(len(heights),len(heights[2500]))

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
    #bpy.ops.view3d.view_selected(use_all_regions=False)

```

## C#

```
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[RequireComponent(typeof(MeshFilter),typeof(MeshRenderer), typeof(MeshCollider))]
public class CoarseMeshGen : MonoBehaviour
{
    float[] bigArr;
    int height = 2499;
    int width = 2500;
    Mesh mesh;
    Vector3[] vertices;
    int[] triangles;
    public int xSize = 250;
    public int zSize = 250;
    public float morphSpeed = 1.0f;
    public int xoff = 0;
    public int zoff = 0;

    [Range(0.0f,1.0f)]
    public float t = 0.0f;
    float lastT = -0.0f;
    ComputeBuffer positionsBuffer;
    ComputeBuffer positionsBuffer2;
    ComputeBuffer outputPositions;
    ComputeBuffer fftBuffer;
    Vector3[] buff;
    Vector3[] buff2;
    public Material mat;
    [SerializeField]
    ComputeShader cs;
    static readonly int tId = Shader.PropertyToID("_t");

    public AudioPlayer audioPlayer;

    // public enum type {sine,slide};
    public bool sine = false;
    public float audioLerpFactor = 0.0f;
    public float audioMultiplier = 1.0f;

    // public Shader shader;
    // Material material;




    [ContextMenu("Start")]
    void Start(){
        // material = new Material(shader);
        loadMeshAndGenerateBufferArrays();
        positionsBuffer.SetData(buff);
        positionsBuffer2.SetData(buff2);
        cs.SetBuffer(0, "_Positions", positionsBuffer);
        cs.SetBuffer(0, "_Positions2", positionsBuffer2);
        cs.SetBuffer(0, "_OutputPositions", outputPositions);
        // material.SetBuffer(0,"meshVertices", outputPositions);


        GenerateMesh();
        UpdateMesh();
        updateOnGPU();
    }
    void Update(){
        if(t != lastT || audioPlayer.GetComponent<AudioSource>().isPlaying){
            updateOnGPU();
            // Debug.Log("gpu");
            lastT = t;
        }
        if(sine){
        t = (Mathf.Sin(Time.time * morphSpeed) + 1)/2;
        }
        // updateOnGPU();
    }


    [ContextMenu("Generate mesh")]
    void GenerateMesh() 
    {
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;
        CreateShape();
        UpdateMesh();
    }


    void OnEnable () {
        // loadData();
		positionsBuffer = new ComputeBuffer((xSize+1)*(zSize+1), 3*sizeof(float));
        positionsBuffer2 = new ComputeBuffer((xSize+1)*(zSize+1), 3*sizeof(float));
        outputPositions = new ComputeBuffer((xSize+1)*(zSize+1), 3*sizeof(float));
        fftBuffer = new ComputeBuffer(256, sizeof(float));

	}

    void OnDisable () {
		positionsBuffer.Release();
        positionsBuffer = null;
        positionsBuffer2.Release();
        positionsBuffer2 = null;
        outputPositions.Release();
        outputPositions = null;
        fftBuffer.Release();
        fftBuffer = null;
	}
    
    [ContextMenu("Buffer Setup")]
    void bufferSetup(){
                    // heightsBuffer = new ComputeBuffer(width*height,sizeof(float));
            // loadData();
            // heightsBuffer.SetData(bigArr);
            // cs.SetBuffer(0,heightsId,heightsBuffer);
        positionsBuffer = new ComputeBuffer((xSize+1)*(zSize+1), 3*sizeof(float));
        positionsBuffer2 = new ComputeBuffer((xSize+1)*(zSize+1), 3*sizeof(float));
        outputPositions = new ComputeBuffer((xSize+1)*(zSize+1), 3*sizeof(float));
        positionsBuffer2.SetData(buff2);
        positionsBuffer.SetData(buff);
        cs.SetBuffer(0, "_Positions", positionsBuffer);
        cs.SetBuffer(0, "_Positions2", positionsBuffer2);
        cs.SetBuffer(0, "_OutputPositions", outputPositions);
    }
    [ContextMenu("Update on gpu")]
    void updateOnGPU(){
        // if(positionsBuffer == null || positionsBuffer2 == null){
        //     bufferSetup();
        // }
        // positionsBuffer2.SetData(buff2);
        // positionsBuffer.SetData(buff);
        cs.SetBuffer(0, "_Positions", positionsBuffer);
        cs.SetBuffer(0, "_Positions2", positionsBuffer2);
        cs.SetBuffer(0, "_OutputPositions", outputPositions);
        // material.SetBuffer (0, "meshVertices", outputPositions);

        fftBuffer.SetData(audioPlayer.spectrum);
        cs.SetBuffer(0, "_fftBuffer", fftBuffer);

        cs.SetFloat("_audioLerpFactor",audioLerpFactor);
        cs.SetFloat("_audioMultiplier",audioMultiplier);

        int groups = Mathf.CeilToInt(zSize / 8f);
        cs.SetInt("_Groupsize",groups);
        int kernelHandle = cs.FindKernel("CSMain");
        cs.SetFloat("_t",t);
		cs.Dispatch(kernelHandle, groups * groups, 1, 1);
        Vector3[] data = new Vector3[(xSize+1) * (zSize+1)];
        outputPositions.GetData(data);
        // Debug.Log(data[62000]);
        mesh.vertices = data;
        mesh.RecalculateNormals ();
    }

    float[] loadTerrainArray(string filepath, int arraySize){
        float[] temp = new float[arraySize];
        string fpath = Path.Combine(Application.streamingAssetsPath, filepath);
        try
        {
            using (var fileStream = System.IO.File.OpenRead(fpath))
            using (var reader = new System.IO.BinaryReader(fileStream))
            {
                for(int i = 0; i < arraySize; i++){
                    temp[i] = reader.ReadSingle();
                }
                return temp;
            }
        }
        catch(System.Exception e){ // handle errors here.
            Debug.Log(e);
        }
        return new float[1];
    }
    
    [ContextMenu("Load custom mesh data and generate buffers")]
    void loadMeshAndGenerateBufferArrays(){
        int buffSize = (xSize + 1) * (zSize + 1);
        buff = new Vector3[buffSize];
        buff2 = new Vector3[buffSize];

        float[] temp = loadTerrainArray("xsize_zsize_251_step_20_offset_12000.dat",buffSize);
        float[] temp2 = loadTerrainArray("xsize_zsize_251_step_20_offset_18000.dat",buffSize);

        int step = 20;
        
        for (int i = 0, z = 0; z<= zSize * step; z+= step)
        {
            for (int x = 0; x<=xSize * step; x+= step)
            {
                // float y = get_height(xoff + x, zoff + z);
                buff[i] = new Vector3(x, temp[i], z);
                // y = get_height(xoff + xSize + x, zoff + zSize + z);
                buff2[i] = new Vector3(x, temp2[i], z);
                i++;
            }
        }
    }

    void CreateShape()
    {
        vertices = new Vector3[(xSize + 1) * (zSize + 1)];
        for(int i = 0; i < vertices.Length; i++){
            vertices[i] = buff[i];
        }

        // for (int i = 0, z =0; z<= zSize; z++)
        // {
        //     for (int x = 0; x<=xSize; x++)
        //     {
        //         // float y = get_height(xoff + x, zoff + z);
        //         // float y = get_height(x, z);
        //         float y = buff[i];
        //         vertices[i] = new Vector3(x, y, z);
        //         i++;
        //     }
        // }

        triangles = new int[xSize * zSize * 6];

        int vert = 0;
        int tris = 0;

        for (int z = 0; z < zSize; z++)
        {
            for (int x = 0; x < xSize; x++)
            {
                triangles[tris + 0] = vert + 0;
                triangles[tris + 1] = vert + xSize + 1;
                triangles[tris + 2] = vert + 1;
                triangles[tris + 3] = vert + 1;
                triangles[tris + 4] = vert + xSize + 1;
                triangles[tris + 5] = vert + xSize + 2;

                vert++;
                tris += 6;
            }
            vert++;
        }

    }
    void UpdateMesh()
    {
        mesh.Clear();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        // optionally, add a mesh collider (As suggested by Franku Kek via Youtube comments).
        // To use this, your MeshGenerator GameObject needs to have a mesh collider
        // component added to it.  Then, just re-enable the code below.
        
        mesh.RecalculateBounds();
        MeshCollider meshCollider = gameObject.GetComponent<MeshCollider>();
        meshCollider.sharedMesh = mesh;
        
    }
}
```

## compute

```hlsl
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel CSMain

RWStructuredBuffer<float3> _Positions;
RWStructuredBuffer<float3> _Positions2;

float _t;

[numthreads(64,1,1)]
void CSMain (uint id : SV_DispatchThreadID){
    float3 pos1 = _Positions[id];
    float3 pos2 = _Positions2[id];
    float3 newPos = float3(pos1.x,_t * pos1.y + (1 - _t) * pos2.y,pos1.z);
    _Positions[id] = newPos;
}
```





```hlsl
// Each #kernel tells which function to compile; you can have many kernels
#pragma kernel CSMain

RWStructuredBuffer<float3> _Positions;
RWStructuredBuffer<float3> _Positions2;
RWStructuredBuffer<float3> _OutputPositions;
RWStructuredBuffer<float> _fftBuffer;

float _t;
float _audioLerpFactor;
float _audioMultiplier;

// y=amplitude×sin(frequency×time+phase)+bias.

float invLerp(float from, float to, float value){
  return (value - from) / (to - from);
}

[numthreads(64,1,1)]
void CSMain (uint id : SV_DispatchThreadID, uint gid : SV_GROUPID){
    float3 pos1 = _Positions[id];
    float3 pos2 = _Positions2[id];

    // float2 point = float2(pos1.x,pos1.z);
    // float xDist = pow(point.x, 2500.0f);
    // float zDist = pow(point.y, 2500.0f);

    float distFromCenter = distance(float3(pos1.x,0.0f,pos1.z),float3(2500.0f,0.0f,2500.0f));
    float circleRadius = abs(distFromCenter - 2500) / 5000;

    // circleRadius -= frac(circleRadius * 10);
    // _t = sin(_Time.y * speed);
    // _t = sin(_t * 0.1 + circleRadius);

    // uint bin = (uint)lerp(0,255,circleRadius);
    // bin = (uint)(bin + _t) % 255;
    // float audio = log10(_fftBuffer[bin]);

    float f_bin = lerp(0.0f,255.0f,circleRadius);
    uint lower_bin = (uint)(f_bin - frac(f_bin));
    uint higher_bin = (lower_bin + 1) % 255;

    float audioLerpFactor = frac(f_bin);
    audioLerpFactor = 0.5;
    audioLerpFactor = _audioLerpFactor;
    float audio = lerp(log10(_fftBuffer[lower_bin]),log10(_fftBuffer[higher_bin]), audioLerpFactor);
    audio *= _audioMultiplier;
    audio *= invLerp(0.0, 3000.0, distFromCenter);
    

    float newheight = _t * pos1.y + (1 - _t) * pos2.y;
    float affector = saturate(invLerp(newheight, 200.0f, 900.0f));
    // _t = affector;

    // audio *= _t;
    // _t = audio * affector * 40;

    // audio = audio * affector * 10;
    // newheight += audio;

    // newheight = affector * 10;
    audio *= 6.0;
    newheight = newheight + audio * affector;

    float3 newPos = float3(pos1.x,newheight,pos1.z);

    _OutputPositions[id] = newPos;
}


[numthreads(64,1,1)]
void CSMainBackup (uint id : SV_DispatchThreadID, uint gid : SV_GROUPID){
    float3 pos1 = _Positions[id];
    float3 pos2 = _Positions2[id];

    // float2 point = float2(pos1.x,pos1.z);
    // float xDist = pow(point.x, 2500.0f);
    // float zDist = pow(point.y, 2500.0f);

    float distFromCenter = distance(float3(pos1.x,0.0f,pos1.z),float3(2500.0f,0.0f,2500.0f));
    float affected_t = abs(distFromCenter - 2500) / 5000;
    // _t = sin(_Time.y * speed);
    // _t = sin(_t * 0.1 + affected_t);

    // uint bin = (uint)lerp(0,255,affected_t);
    // bin = (uint)(bin + _t) % 255;
    // float audio = log10(_fftBuffer[bin]);

    float f_bin = lerp(0.0f,255.0f,affected_t);
    uint lower_bin = (uint)(f_bin - frac(f_bin));
    uint higher_bin = (lower_bin + 1) % 255;
    float audio = lerp(log10(_fftBuffer[lower_bin]),log10(_fftBuffer[higher_bin]), frac(f_bin));
    

    float newheight = _t * pos1.y + (1 - _t) * pos2.y;
    float affector = saturate(invLerp(newheight, 200.0f, 900.0f));
    // _t = affector;

    // audio *= _t;
    // _t = audio * affector * 40;

    audio = audio * affector * 10;
    newheight += audio;


    // _t *= affector;
    // _t = _t * affected_t * (log10(_fftBuffer[bin]));
    // _t = saturate(_t) * 2 - 1;
    // newheight += _t;
    // float3 newPos = float3(pos1.x,_t * pos1.y + (1 - _t) * pos2.y,pos1.z);
    float3 newPos = float3(pos1.x,newheight,pos1.z);
    // newPos = _Positions[id];
    // newPos.y = affected_t;
    // _Positions[id] = newPos;
    _OutputPositions[id] = newPos;
}
```