# RayOfHope

An attempt to creating a digital twin of a camera inside a digital eleveation model of the earth constructed from open access JAXA data. The goal is to be able to geolocate object in the cameras view and then beeing able to estimate the possition of the object using noithing but knwon local parameter of the camera.


**The first step is to take the geotiff file.**

<img src="https://github.com/ttsiapras/RayOfHope/blob/96906466bd01f08b7f3e0b55bec08c69e5a889f5/imgs/Figure.png" alt="drawing" style="width:500px;"/>

**And create 3d mesh or surface so we can have a difital twin of our enviroment.**

<img src="https://github.com/ttsiapras/RayOfHope/blob/96906466bd01f08b7f3e0b55bec08c69e5a889f5/imgs/Figure3d.png" alt="drawing" style="width:500px;"/>
<img src="https://github.com/ttsiapras/RayOfHope/blob/96906466bd01f08b7f3e0b55bec08c69e5a889f5/imgs/open3d.png" alt="drawing" style="width:500px;"/>



## Roadmap

- Create georeferncing pipeline so each point can have a known Lat/Long.

- Implement routine to fine intersection between a knwon line expression and the surface.

- Take camera parameters and create appropriate line in the 3d space.


## Authors

- [@ttsiapras](https://github.com/ttsiapras)

## Dataset resources

* https://glovis.usgs.gov/app : OrbView3 satellite. Pan 1m / MS 4m
* https://browser.dataspace.copernicus.eu 
