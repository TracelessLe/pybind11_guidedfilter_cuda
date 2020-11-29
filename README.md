# pybind11_guidedfilter_cuda
Use pybind11 to generate a python binding of guided filter with cuda enabled.

# install
1. git clone
2. cd into the project path
3. cmake .
4. make
5. use it

# use in python
```
import cv2
import gfcuda

img = cv2.imread('cat.png')
guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
final_img = gfcuda.guidedFilter(guide=guide, src=img, radius=5, eps=50*50, dDepth=-1)
cv2.imwrite('out.png', final_img)
```
