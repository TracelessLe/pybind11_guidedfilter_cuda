import cv2
import time
import numpy as np
import gfcuda

img = cv2.imread('cat.png')
guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

total_count = 1000
start_time = time.time()
cv2.cuda.setDevice(0)
for i in range(total_count):
    iter_start_time = time.time()
    final_img = gfcuda.guidedFilter(guide=guide, src=img, radius=5, eps=50*50, dDepth=-1)
    print('iter gfcuda cost: ', i,  time.time() - iter_start_time)
cost_time = time.time() - start_time
print('cuda_guidedFilter cost: ', cost_time)
cv2.imwrite('out.png', final_img)

start_time = time.time()
for i in range(total_count):
    iter_start_time = time.time()
    final_img = cv2.ximgproc.guidedFilter(guide=guide, src=img, radius=5, eps=50*50, dDepth=-1)
    print('iter cpu-gf cost: ', i,  time.time() - iter_start_time)
cost_time = time.time() - start_time
print("cpu_guidedFilter cost: ", cost_time)
