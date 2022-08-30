# pybind11_guidedfilter_cuda
Use pybind11 to generate a python binding of guided filter with cuda enabled.

# Dependencies
1. OpenCV with CUDA enabled
2. Python
3. Pybind11

# Installation
1. git clone https://github.com/TracelessLe/pybind11_guidedfilter_cuda.git
2. cd into the project path
3. git clone https://github.com/pybind/pybind11.git
4. cmake .
5. make
6. use it

# Tutorials
Use in Python
```

import cv2
import gfcuda

img = cv2.imread('cat.png')
guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
final_img = gfcuda.guidedFilter(guide=guide, src=img, radius=5, eps=50*50, dDepth=-1)
cv2.imwrite('out.png', final_img)

```

![input](https://github.com/TracelessLe/pybind11_guidedfilter_cuda/raw/master/cat.png) ![output](https://github.com/TracelessLe/pybind11_guidedfilter_cuda/raw/master/out.png)
