---
title: "Simple 1D Quantum Tunneling in Python"
---

<meta charset="UTF-8">
<script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$']]
        }
    };
</script>
<script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


```python
import numpy as np
from scipy import interpolate as scp_interpolate
import scipy.stats as st
from manim import *
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Manim Community <span style="color: #008000; text-decoration-color: #008000">v0.17.3</span>

</pre>



#### Manim Configuration


```python
config.media_width = "75%"
config.verbosity = "WARNING"
```

# Simple 1D Quantum Tunneling in Python
## Animated in Manim


I was visiting my parents for the holidays and found some old class notes from introductory quantum mechanics, a class that only began to make sense in retrospect. Since then, I have gained some programming skills, so I set out to see if I could numerically solve some simple 1-dimensional systems. It turned out to be successful, and using Manim to make the animation shine was fun. There are several other versions of this online, but I found this one from Dot Physics to be the most informative. Even still, for many of the examples I saw, the code reads differently than the math a student encounters, which is fine for the program but not for the student. Here, I implement a Gaussian wave and its interaction with a barrier to draw a clear link between the math used and the code employed.

A big caveat to the pedalogical goal of this project is that I explain nothing here, I only demonstrate.

### Define $x$ and construct $\langle x |$ and $dx$


```python
J = 1000  # Size of the position mesh.
x = np.linspace(-2, 10, J + 1)
dx = x[1] - x[0]  # Step-size of the position mesh.
```

### Define Constants and Set Parameters

See the publication for relationships you can use to estimate parameters.


```python
m = 1
ħ = 1
```

Here I get a momentum value that will be reasonable to capture.


```python
# Parameter estimations.
N_time_steps = 24 * 5
time_step = 5 / 24
λ = 2 * dx**2 / time_step
k_0 = J * λ / 8 * N_time_steps
k_0 = np.round(k_0)
k_0
```




    21.0



### Define a barrier energy upon $\hat{V} = \langle x | V(x)$


```python
V_0 = k_0**2/(2*m)
# V_0 = 2*(50*np.pi)**2
# V_0 = 10 / 2 * J * dx
V = np.zeros_like(x)
V[(5 < x) & (x < 6)] = V_0
```

### Create the operator matrix $\hat{T}$, then calculate $\hat{H} = \hat{T} + \hat{V}$


```python
T = (  # Time-independent kinetic energy operator of the system.
      ( ħ**2/(  m*dx**2)) * np.diag(np.ones(J-1)) 
    + (-ħ**2/(2*m*dx**2)) * np.diag(np.ones(J-2), 1)
    + (-ħ**2/(2*m*dx**2)) * np.diag(np.ones(J-2),-1)
)

# Here V depends only on the current position.
H = T + np.diag(V[1:-1])
```

### Calculate $E$ and $\psi$ by taking the eigen values and vectors of $\hat{H}$


```python
E, ψ = np.linalg.eigh(H)
ψ = ψ.T
```

### Define $\psi(x,t)$ and $\psi(x,0)$, then calculate $| \psi_0 \rangle = \langle x | \Psi (x,0)$

I use a gaussian wavefunction:

$g(x) = e^{-\frac{1}{2}\frac{(x - x_0)^2}{\sigma^2}}$

It is wild we can do so much without considering this function.


```python
x_0 = 2.5   # Center of the wavefunction.
σ = 0.5     # Width of the wavefunction.
ψ_x = x[1:-1]  # So that we dont differentiate the edge points.
ψ_0 = np.exp(-1/2*(ψ_x - x_0)**2 / σ**2) * np.exp(1j*k_0*ψ_x)
```

### Calculate $A = \sum \psi_0^* \psi_0 dx$, then Normalize $\Psi = \frac{\psi}{\sqrt{A}}$


```python
# Calculate and apply normalization factor.
A = np.sum(ψ[0] * np.conj(ψ[0]) * dx)
ψ = ψ / np.sqrt(A)
```

### Calcualte $c = \langle \psi^* | \Psi_0 \rangle dx$


```python
c = np.conj(ψ) @ ψ_0 * dx
```

### Calculate $\Psi(t)$


```python
def psi_t(t):
    return ψ.T @ (c*np.exp(-1j*E*t/ħ))
```

## Animate with Manim


```python
%%manim -qh -o quantum_tunneling.mp4 --format mp4 --media_dir manim_output Moving_Wavefunction

class Moving_Wavefunction(Scene):
    def construct(self):
        axes = Axes(x_range=[-2, 10, 1], y_range=[-2, 2.0, 0.5], tips=False)
        axes_labels = axes.get_axis_labels(Tex("x").scale(0.7), Text("P").scale(0.45))
        # Construct the barrier as a polygon is easier than messing with step functions.
        # What the height means here is up to you.
        barrier_verts = [axes.c2p(*v) for v in [[5, 0, 0], [5, 1, 0], [6, 1, 0], [6, 0, 0]]]
        barrier = Polygon(*barrier_verts, color=RED, fill_opacity=0.1)
        time_tracker = ValueTracker(0)
                
        def plot_ψ_t():
            x_range = (0., 9.9, 0.01)
            curr_ψt = psi_t(time_tracker.get_value())
            curr_ψ_interp = scp_interpolate.interp1d(ψ_x, curr_ψt)
            psi_abs  = axes.plot(lambda x: np.abs(curr_ψ_interp(x)), x_range=x_range, color=WHITE)
            psi_real = axes.plot(lambda x: np.real(curr_ψ_interp(x)), x_range=x_range, color=ORANGE, stroke_width=1)
            psi_imag = axes.plot(lambda x: np.imag(curr_ψ_interp(x)), x_range=x_range, color=BLUE, stroke_width=1)
            return VGroup(psi_real, psi_imag, psi_abs)

        graph = always_redraw(plot_ψ_t)
        self.play(Create(VGroup(axes, axes_labels, graph)))
        self.play(Create(barrier))
        self.play(time_tracker.animate.set_value(0.25), run_time=20,
                  rate_func=rate_functions.linear)
        self.play(*[FadeOut(mob)for mob in self.mobjects])
```

                                                                                                                           

<video src="./assets/quantum_tunneling/quantum_tunneling.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>



**Resources and References**

+ [Computer-Generated Motion Pictures of One-Dimensional Quantum-Mechanical Transmission and Reflection Phenomena ](https://pubs.aip.org/aapt/ajp/article-abstract/35/3/177/1042551/Computer-Generated-Motion-Pictures-of-One?redirectedFrom=fulltext)
+ [Manim Documentation](https://docs.manim.community/en/stable/index.html)
+ [FlyingFrames](https://flyingframes.readthedocs.io/en/latest/index.html)
+ [Dot Physics Quantum Tunneling](https://www.youtube.com/watch?v=j8cjzZG1qa8&t=525s)
