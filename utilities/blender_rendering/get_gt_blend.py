import bpy
import os

import sys

bpy.ops.wm.open_mainfile(filepath = sys.argv[-1])

bpy.ops.object.editmode_toggle()
bpy.ops.curve.select_all(action='SELECT')
bpy.ops.curve.subdivide(number_cuts=10)

currPath = os.path.splitext(bpy.data.filepath)[0]+".txt"
file = open(currPath, "w") 
print(currPath)
for curves in bpy.data.curves:
    for splines in curves.splines:
        for x in range(len(splines.bezier_points)): 
            file.write("%.3f " % (splines.bezier_points[x].co.x)) 
            file.write("%.3f " % (splines.bezier_points[x].co.y)) 
            file.write("%.3f " % (splines.bezier_points[x].co.z))
            file.write("\n")

file.close()
