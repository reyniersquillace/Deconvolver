<h1>Overview</h1>

Welcome to my collection of neural network methods for the decomposition of complex pulse profiles! There are a variety of trained models here suitable for different applications. The models themselves are stored in the `pickle_jar`, where the `.pkl` files live along with the relevant loss curves. Along with the models, this repo contains code for training the models yourself. 

This code was designed with the decomposition of multi-pulse radio transient profiels in mind. It works just about as well as `scipy.signal.find_peaks`, but you should probably just use <a href = "https://iopscience.iop.org/article/10.3847/1538-4357/ad1ce7/meta">CLEAN</a>.

<h1> Dependencies </h1>


<li> numpy </li>
<li> matplotlib </li>
<li> sklearn </li>
<li> pytorch </li>
<li> optuna </li>

<h1>Usage</h1>
In order to use a model, clone this repo. Then, in your command line, type:

`python3 Call.py [./path/to/model] use [./path/to/pulse/profile]`

I have uploaded the three (real) pulse profiles I used as my test cases to the directory `tests`. 

If you want to generate your own profiles to see how well the model lines up with what you feed into it, then type:

`python3 Call.py [./path/to/model] test [n_test]`

where `n_test` is the number of pulses you want generated. This will give you information about the model accuracy. 

Unfortunately, working on this project against a deadline means I haven't had time to train all the models I wanted to. But fear not, my code allows you to train your own using my architecture and modules! To see how that works, head on over to the repo wiki.
