# https://stackoverflow.com/questions/45179024/scipy-bspline-fitting-in-python
import random
import bezier
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import os
from tqdm import tqdm
from PIL import Image
import copy 

np.random.seed(123)
random.seed(123)

dest = './data/bspline_fc_1_pID/train'
num_ids = 247

samples_per_id = 1 # > 1 to sample raw CPD data
                   # = 1 to sample one image per ID; use ./augs.py to sample raw VPD data

for i in tqdm(range(1, num_ids+1)):

    mask = random.choices([True, False], [0.4, 0.6], k=random.randint(3, 6))

    # None padding
    if len(mask) == 3:
        mask.insert(0, None)
        mask.insert(0, None)
        mask.append(None)

    if len(mask) == 4:
        mask.insert(0, None)
        mask.append(None)

    if len(mask) == 5:
        mask.append(None)

    grid_size = 12
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
 
    ax.set_aspect('equal')
    ax.axis("off")

    global_arr = [] 
    counter = 0
    for idx, val in enumerate(mask):
        start_delta_x = np.random.uniform(0, 1)
        end_delta_x = np.random.uniform(0, 1)
        
        if val == True:
            
            x1 = 1 + start_delta_x 
            x2 = 11 - end_delta_x
            y1 = 2 * idx + 1 
            y2 = 2 * idx + 1 
            
            degree = random.choice([3, 4])
        
            if degree == 4:
                x = [x1]
                y = [y1]
                for j in range(1, degree+1):
                    x.append(x1 + 2 * j)
                    y.append(y1 + np.random.uniform(-0.4, 0.4))
                x.append(x2)
                y.append(y2)

            elif degree == 3:
                x = [x1]
                y = [y1]
                for j in range(1, degree+1):
                    x.append(x1 + 2.5 * j)
                    y.append(y1 + np.random.uniform(-0.4, 0.4))
                x.append(x2)
                y.append(y2)
                
            x = np.array(x)
            y = np.array(y)
            global_arr.append([x, y, degree])
            t, c, k = interpolate.splrep(x, y, s=0, k=4)
            
            N = 100
            xmin, xmax = x.min(), x.max()
            xx = np.linspace(xmin, xmax, N)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
  
            ax.plot(xx, spline(xx), 'w', linewidth = np.random.uniform(3, 4), label='BSpline')

        if val == False:
            wrinkle_mask = random.choices([True, None], [0.4, 0.6], k=3)
            
            for idx_w, w in enumerate(wrinkle_mask):
                wstart_delta = np.random.uniform(0, 0.5)
                wend_delta = np.random.uniform(0, 0.5)

                if w == True:
                    wx1 = 4 * idx_w + wstart_delta
                    wy1 = 2 * idx + 1
                    wx2 = 4 * idx_w + 4 - wend_delta
                    wy2 = 2 * idx + 1

                    wdegree = 2
                    if wdegree == 2:
                        mid_x = (wx1 + wx2)//2
                        mid_y = (wy1 + wy2)//2 + random.choice([-0.3, 0.3])

                        xw = [wx1 + 0.4, mid_x - 0.4, wx2]
                        yw = [wy1, mid_y, wy2]
                        control_points = np.array([xw, yw])

                        global_arr.append([xw, yw, wdegree])
                        curve = bezier.Curve(control_points, degree = wdegree)
                        s_vals = np.linspace(0.0, 1.0, 100)
                        curve_points = curve.evaluate_multi(s_vals)
                        ax.plot(curve_points[0, :], curve_points[1, :], 'w', linewidth = np.random.uniform(3, 4))

                if w == None:
                    continue

        if val == None:
            continue
        
    plt.style.use('dark_background')

    dest_i = f'{dest}/{i}'
    os.makedirs(dest_i, exist_ok=True)
    # plt.tight_layout()
    plt.savefig(f'{dest_i}/1.png')
    plt.close()

    ### add intra-class diversity if samples_per_id > 1 ###
    for num in range(2, samples_per_id+1):
        
        grid_size = 12
        fig, ax2 = plt.subplots(figsize=(6, 6))
        ax2.set_aspect('equal')
        ax2.axis("off")

        for i2 in global_arr:
            x, y, d = copy.deepcopy(i2)
            if d == 3 or d == 4:
                for i3, _ in enumerate(x):
                    if i3 == len(x)//2 or i3 == 0 or i == len(x) - 1:
                        x[i3] = x[i3] + np.random.uniform(-0.3, 0.3)
                        continue
                    else:
                        y[i3] = y[i3] + np.random.uniform(-0.4, 0.4)

                N = 100
                t, c, k = interpolate.splrep(x, y, s=0, k=4)
                xmin, xmax = x.min(), x.max()
                xx = np.linspace(xmin, xmax, N)
                spline = interpolate.BSpline(t, c, k, extrapolate=False)
            
                ax2.plot(xx, spline(xx), 'w', linewidth = np.random.uniform(3, 4), label='BSpline')

            if d == 2:
                for i3, _ in enumerate(x):
                    if i3 == len(x)//2 or i3 == 0 or i3 == len(x) - 1:
                        x[i3] = x[i3] + np.random.uniform(-0.2, 0.2)
                        continue
                    else:
                        y[i3] = y[i3] + np.random.uniform(-0.5, 0.5)
                        
                curve = bezier.Curve([x, y], degree = d)
                s_vals = np.linspace(0.0, 1.0, 100)
                curve_points = curve.evaluate_multi(s_vals)
                ax2.plot(curve_points[0, :], curve_points[1, :], 'w', linewidth = np.random.uniform(3, 4))

        plt.savefig(f'{dest_i}/{num}.png')
        plt.close()
