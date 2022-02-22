# Graphics

## Assignment2 - basic transformation of 3d graphics


> Data is given in ply file format containing verticies(vertex coordinations) and faces(edge coordinates)

<div align="center">
   <a href="">
     <img src="img/assignment2/cube.gif" alt="img" width="380" height="380">
   </a>
   <p>Figure1. Cube resize, rotation, move position</p>
   </br>
   <a href="">
     <img src="img/assignment2/polygon.gif" alt="img" width="380" height="380">
   </a>
   <p>Figure2. Polygon resize, rotation, move position</p>
</div>


## Assignment3 - Morphing and Delaunay triangulation with faces


> Data given with .pts format
<div align="center">
   <a href="">
     <img src="output_1.gif" alt="img" width="480" height="380">
   </a>
   <p>Figure1. First attempt, using only average vertex points</p>
   <p>Used Python Delaunay from scipy.spatial package</p>
   <p>
      Problem1 - Edges change on morph. Needs to stay along with targeted vertices.<br>
      Problem2 - Y axis possibly flipped/\.
   </p>
   </br>
</div>

> Data given with .asf format
<div align="center">
   <a href="">
     <img src="img/assignment2/movie.gif" alt="img" width="500" height="500">
   </a>
   <p>Figure2. Final attempt, used vector multiplication on triangle edges</p>
   <p>Used Python Delaunay from scipy.spatial package for initial search</p>
   </br>
</div>

