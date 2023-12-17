# Tyler Biggs' Website

Welcome, my name is Tyler. I began programming during my doctoral research in synthetic organic chemistry,
and have now transitioned into chem/bioinformatics.

This site is a small collection of 'data science' projects that I found interesting, and I hope you do as well.

> For the things we have to learn before we can do them, we learn by doing them.
> -Aristotle


## Projects

### [***Learning Programming with Conway's Game of Life***](./docs/raytrace_gol)

I began work on a short course or module designed to introduce programming for data science to 
graduate students. Preparing material for students with such a wide range of potential experiences is challenging,
and as the development of the course was cancelled, I never was able to polish the course to any degree. That said, 
the capstone project for the students produces some lovely ray-traced images, and the method we use to permute each 
step of the game using a convolution is a good step from a programming point of view to a mathematical and scientific one.

<video src="/docs/assets/gol_HDr.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>



[***Exploring Stochastic Simulations with a Seminal Paper***](./docs/gillespie)

Here I implement the algorithm described in *Exact Stochastic Simulation of Coupled Chemical Reactions* by Danlel T. Gillespie. It is a seminal paper and well worth the read, a copy can be found online [here](https://www.caam.rice.edu/~cox/gillespie.pdf). Much of the code below borrows notation from the paper. Following the paper, for testing our algorithm I use [exponential decay](https://en.wikipedia.org/wiki/Exponential_decay). Such processes are ubiquitous in nature, so the example is worthwhile. Additionally we can numerically solve many features of interest in this system, and compare those features to our simulation results.

<video src="/docs/assets/gillespie/gillespie.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>