# Ray Tracing a 3D Version of Conway's Game of Life

Art is not a separate skill from science but part of its output. Here I demonstrate the use of `plotoptix`, a ray-tracing engine with 
a 3D implementation of the Game of Life. While ray-tracing makes everything look flashy, the most interesting piece here is 
the use of a convolution for each generation (see the `step` function).

Using only the code in this notebook, we can generate animations like the following:


<video src="./assets/gol_HDr.mp4" controls  width="500" >
Your browser does not support the <code>video</code> element.
</video>


***Preamble***


```python
import numpy as np
from numpy.random import default_rng
import cv2
from scipy.signal import convolve
from scipy.special import expit

rng = default_rng()

import matplotlib.pyplot as plt
%matplotlib inline

# from plotoptix import TkOptiX
from plotoptix import NpOptiX
from plotoptix.materials import m_clear_glass, m_mirror, m_plastic
from plotoptix.utils import map_to_colors
```

## Game of Life Simulation

This code is the product of a lesson series. While I never found the time to polish it to my liking, you can see the notebooks 
[here](https://github.com/biggstd/teaching_datasci/tree/main/game_of_life). The idea wasn't that the students would do this, 
rather they would take their completed game of life with the same class structure (duck-typed) as my example and could create a nice render.

For now, we will use the code as is:


```python
class GoLSimulation:

    def __init__(self, grid):
        self.grid = grid.astype(int)
        self.grid_history = list([grid])

    def step(self, x):
        conv_array = np.ones((3, 3, 3))
        nbrs_count = convolve(x, conv_array, mode='same') - x
        return ((nbrs_count >= 3) | (x & (nbrs_count >= 2))) & (nbrs_count <= 5)

    def simulate(self, n_iter:int):
        for _ in range(n_iter):
            new_grid = self.step(self.grid)
            self.grid_history.append(new_grid)
            self.grid = new_grid
        return self

    def _repr_png_(self):
        plt.figure(figsize=(5,5)).add_subplot(projection='3d').voxels(self.grid)
```

Testing our simulation.


```python
grid_3d = np.zeros((25, 25, 25))
random_pop = (rng.random(grid_3d.shape) > 0.95).astype(int)

gol_sim = GoLSimulation(grid=random_pop)
gol_sim = gol_sim.simulate(5)
gol_sim
```




    <__main__.GoLSimulation at 0x2ae52199bd0>




<image src="./assets/output_8_1.png"  width="500" >
output image
</image>

    


Rather than randomly seeding starting cells, I want to have a text block.
Perhaps we can even find a trick to have the blocks evolve into text.


```python
image = np.zeros((100, 150, 3), np.uint8)
position = (30, 60)  # Fiddle with this to center your text.

cv2.putText(image, "BIGGS",
            position,
            cv2.FONT_HERSHEY_PLAIN, #font family
            2, #font size
            (255, 255, 255, 255), #font color
            3)
text_image = (image[:, :, 0] > 0).astype(int)
text_grid = np.zeros((text_image.shape[0], text_image.shape[1], 16), dtype=int)
for idx in range(7, 8):
    text_grid[:, :, idx] += text_image

plt.imshow(text_image, cmap='Greens');
```


  
<image src="./assets/output_10_0.png"  width="500" >
output image
</image>

    


Now, I provide the text grid made above as the input.


```python
%%time
gol_sim = GoLSimulation(grid=text_grid)
gol_sim = gol_sim.simulate(8)
```

    CPU times: total: 109 ms
    Wall time: 114 ms
    

I don't want to render static cubes for the animation. I want the cubes to change color as they shrink or grow out of and into existance. The following function creates the sizes, colors for this between generation animation.


```python
def size_flux(gen, n_frames, sim=gol_sim):
    """For each generation color and transform the cell based on its status."""
    if gen == 0:
        prev_pos = np.zeros_like(sim.grid_history[0])
    else:
        prev_pos = sim.grid_history[gen - 1]
        
    curr_pos = sim.grid_history[gen]
    
    stable_cells = np.argwhere((prev_pos == 1) & (curr_pos == 1))
    new_cells    = np.argwhere((prev_pos == 0) & (curr_pos == 1))
    dying_cells  = np.argwhere((prev_pos == 1) & (curr_pos == 0))
    
    stable_cell_sizes = (1 - np.abs(np.sin(np.linspace(0, np.pi, n_frames)[:, np.newaxis])) * 0.15 * np.ones(stable_cells.shape[0]))
    
    starts = rng.choice(np.arange(4, 6), new_cells.shape[0])
    stops = rng.choice(np.arange(n_frames - 4, n_frames), new_cells.shape[0])
    xx = np.linspace(starts, stops, n_frames) - (n_frames/2)
    new_cell_sizes = expit(xx)

    starts = rng.choice(np.arange(4, 6), dying_cells.shape[0])
    stops = rng.choice(np.arange(n_frames - 4, n_frames), dying_cells.shape[0])
    xx = np.linspace(stops, starts, n_frames) - (n_frames/2)
    dying_cell_sizes = expit(xx)

    pos = np.vstack((stable_cells, new_cells, dying_cells))
    sizes = np.hstack((stable_cell_sizes, new_cell_sizes, dying_cell_sizes))
    return pos, sizes
```

With that function prepared, we can begin using `plotoptix`. Here I add some dynamic camera motion.


```python
class params:
    # Simulation parameters.
    sim = gol_sim
    n_gen = 8
    gen_time = 2
    fps = 30
    width = 1920
    height = 1080
    frame = 0
    
    # Set the cube sizes.
    max_scale = 0.95
    u = np.array([1, 0, 0]) * 1 /24
    v = np.array([0, 1, 0]) * 1 /24
    w = np.array([0, 0, 1]) * 1 /24
    
    # Add some color and movement to the cells.
    pos, gen_uvw = size_flux(0, gen_time * fps)
    colors = map_to_colors(pos[:, 1], "brg")
    
    # Add some fancy camera jitter.
    eye = np.array([3.0, 0.25, 2.5]) * 70
    camera_x = eye[0]
    camera_z = eye[2]
    _tt = np.linspace(0, 2*np.pi, num=gen_time * fps * n_gen)
    camera_xx = 3 * np.cos(_tt) + camera_x
    camera_zz = 8 * np.sin(_tt) + camera_z
    ffov = np.linspace(16, 20, gen_time * fps * n_gen)
    fov = 16
```

Now, I set up some other parameters for the ray tracer and some callbacks. This code is based on the 
[examples provided](https://github.com/rnd-team-dev/plotoptix/tree/master/examples/2_animations_and_callbacks) by the plotoptix team.

The major difference here is that the compute step checks if it is out of the desired frame range and pauses itself if so. 
This is needed. Otherwise, the ray tracer keeps running and calling the compute step.


```python
def init(rt):
    rt.set_param(min_accumulation_step=16, max_accumulation_frames=500)
    rt.set_float("tonemap_exposure", 1.15)
    rt.set_float("tonemap_gamma", 1.0)
    rt.set_float("denoiser_blend", 0.15)
    rt.add_postproc("Denoiser")
    rt.add_postproc("Gamma")
    rt.set_background(0.99)
    rt.set_ambient(0.85)
    rt.setup_material("plastic", m_plastic)
    rt.set_data("plot", pos=params.pos, u=params.u, v=params.v, w=params.w,
                geom="Parallelepipeds", c=params.colors, mat="plastic")
    # Add a background plane so we can see rays reflections.
    rt.set_data("plane", geom="Parallelograms", c=0.99,
                pos=[-1000, 0, -1], u=[5000, 0, 0], v=[0, 1000, 0], w=[0, 0, 1000])
    rt.setup_camera("cam1", cam_type="ThinLens", eye=params.eye,
                    target=[50, 75, 6], up=[0, 0, 1], fov=18, focal_scale=.95)
    

def compute(rt: NpOptiX, delta: int) -> None:
    gen = params.frame // (params.gen_time * params.fps)
    gen_frame = params.frame % (params.gen_time * params.fps)
    
    if params.frame % (params.gen_time * params.fps) == 0:
        # Update generation sizes.
        pos, gen_uvw = size_flux(gen, params.gen_time * params.fps)
        params.pos = pos
        params.gen_uvw = gen_uvw
    
    scale_broadcast = params.gen_uvw[gen_frame, :][:, np.newaxis] * params.max_scale
    
    params.camera_x = params.camera_xx[params.frame]
    params.camera_z = params.camera_zz[params.frame]
    
    params.u = np.array([1, 0, 0]) * scale_broadcast
    params.v = np.array([0, 1, 0]) * scale_broadcast
    params.w = np.array([0, 0, 1]) * scale_broadcast
    params.colors = map_to_colors(params.pos[:, 1], "brg")
    params.fov = params.ffov[params.frame]
    params.frame += delta
    if params.frame >= int(np.floor((params.gen_time * params.fps * params.n_gen))):
        rt.pause_compute()
        rt.encoder_stop()
        rt.close()

def update(rt: NpOptiX) -> None:
    # Optional frame output saving.
    # if params.frame % params.fps == 0:
    #     print(f'frame: {params.frame}')
    #     rt.save_image("output/frame_{:05d}.png".format(params.frame))
    rt.update_data("plot", pos=params.pos, u=params.u, v=params.v, w=params.w, c=params.colors)
    rt.update_camera("cam1", eye=[params.camera_x , 35, params.camera_z], fov=params.fov)      

optix = NpOptiX(on_initialization=init,
                on_scene_compute=compute,
                on_rt_completed=update,
                width=params.width, 
                height=params.height,
                start_now=False)

optix.start()
optix.encoder_create(fps=params.fps, profile="High")
optix.encoder_start("gol_HD.mp4", params.gen_time * params.fps * params.n_gen)
```


```python
print(optix.encoding_frames(), optix.encoded_frames())
```

    480 0
    

After that is done, we can look at our hard work.

<video src="./assets/gol_HD.mp4" controls  width="500" >
Your browser does not support the <code>video</code> element.
</video>




Now for the really hard part: finding a way to make the simulation form into text.
That would be very difficult. Instead, I promised a trick, so let's reverse the video we already made.


```bash
%%bash
ffmpeg -i 'gol_HD.mp4' -vf reverse 'gol_HDr.mp4'
```

    ffmpeg version 5.1.2 Copyright (c) 2000-2022 the FFmpeg developers
      built with clang version 15.0.2
      configuration: --prefix=/d/bld/ffmpeg_1666357623563/_h_env/Library --cc=clang.exe --cxx=clang++.exe --nm=llvm-nm --ar=llvm-ar --disable-doc --disable-openssl --enable-demuxer=dash --enable-hardcoded-tables --enable-libfreetype --enable-libfontconfig --enable-libopenh264 --ld=lld-link --target-os=win64 --enable-cross-compile --toolchain=msvc --host-cc=clang.exe --extra-libs=ucrt.lib --extra-libs=vcruntime.lib --extra-libs=oldnames.lib --strip=llvm-strip --disable-stripping --host-extralibs= --enable-gpl --enable-libx264 --enable-libx265 --enable-libaom --enable-libsvtav1 --enable-libxml2 --enable-pic --enable-shared --disable-static --enable-version3 --enable-zlib --pkg-config=/d/bld/ffmpeg_1666357623563/_build_env/Library/bin/pkg-config
      libavutil      57. 28.100 / 57. 28.100
      libavcodec     59. 37.100 / 59. 37.100
      libavformat    59. 27.100 / 59. 27.100
      libavdevice    59.  7.100 / 59.  7.100
      libavfilter     8. 44.100 /  8. 44.100
      libswscale      6.  7.100 /  6.  7.100
      libswresample   4.  7.100 /  4.  7.100
      libpostproc    56.  6.100 / 56.  6.100
    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'gol_HD.mp4':
      Metadata:
        major_brand     : isom
        minor_version   : 512
        compatible_brands: isomiso2avc1mp41
        encoder         : Lavf58.20.100
      Duration: 00:00:15.90, start: 0.000000, bitrate: 1925 kb/s
      Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 1924 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
        Metadata:
          handler_name    : VideoHandler
          vendor_id       : [0][0][0][0]
    Stream mapping:
      Stream #0:0 -> #0:0 (h264 (native) -> h264 (libx264))
    Press [q] to stop, [?] for help
    [libx264 @ 000001A672250780] using SAR=1/1
    [libx264 @ 000001A672250780] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2 AVX512
    [libx264 @ 000001A672250780] profile High, level 4.0, 4:2:0, 8-bit
    [libx264 @ 000001A672250780] 264 - core 164 r3095 baee400 - H.264/MPEG-4 AVC codec - Copyleft 2003-2022 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=24 lookahead_threads=4 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
    Output #0, mp4, to 'gol_HDr.mp4':
      Metadata:
        major_brand     : isom
        minor_version   : 512
        compatible_brands: isomiso2avc1mp41
        encoder         : Lavf59.27.100
      Stream #0:0(und): Video: h264 (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 30 fps, 15360 tbn (default)
        Metadata:
          handler_name    : VideoHandler
          vendor_id       : [0][0][0][0]
          encoder         : Lavc59.37.100 libx264
        Side data:
          cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
    frame=  477 fps=188 q=-1.0 Lsize=    5566kB time=00:00:15.80 bitrate=2885.6kbits/s speed=6.21x    
    video:5559kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.116329%
    [libx264 @ 000001A672250780] frame I:4     Avg QP:13.62  size: 64726
    [libx264 @ 000001A672250780] frame P:126   Avg QP:18.06  size: 27996
    [libx264 @ 000001A672250780] frame B:347   Avg QP:23.24  size:  5491
    [libx264 @ 000001A672250780] consecutive B-frames:  2.5%  0.8%  1.9% 94.8%
    [libx264 @ 000001A672250780] mb I  I16..4: 54.1% 35.7% 10.2%
    [libx264 @ 000001A672250780] mb P  I16..4:  2.8%  2.1%  1.0%  P16..4: 11.5%  6.5%  3.8%  0.0%  0.0%    skip:72.4%
    [libx264 @ 000001A672250780] mb B  I16..4:  0.1%  0.1%  0.1%  B16..8: 13.4%  2.1%  0.7%  direct: 0.9%  skip:82.6%  L0:36.9% L1:58.4% BI: 4.8%
    [libx264 @ 000001A672250780] 8x8 transform intra:35.9% inter:43.0%
    [libx264 @ 000001A672250780] coded y,uvDC,uvAC intra: 25.3% 43.4% 24.7% inter: 4.4% 6.4% 3.2%
    [libx264 @ 000001A672250780] i16 v,h,dc,p: 45% 19%  3% 34%
    [libx264 @ 000001A672250780] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 38% 21% 25%  2%  2%  5%  2%  2%  5%
    [libx264 @ 000001A672250780] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 25% 20% 16%  4%  5% 14%  4%  3%  9%
    [libx264 @ 000001A672250780] i8c dc,h,v,p: 68% 16% 10%  6%
    [libx264 @ 000001A672250780] Weighted P-Frames: Y:5.6% UV:5.6%
    [libx264 @ 000001A672250780] ref P L0: 56.1% 19.9% 17.5%  6.4%  0.1%
    [libx264 @ 000001A672250780] ref B L0: 86.6% 11.2%  2.2%
    [libx264 @ 000001A672250780] ref B L1: 93.3%  6.7%
    [libx264 @ 000001A672250780] kb/s:2863.79
