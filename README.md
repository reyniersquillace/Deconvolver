Welcome to my collection of neural network methods for the decomposition of complex pulse profiles! There are a variety of trained models here suitable for different applications. The models themselves are stored in the `pickle_jar`, where the `.pkl` files live along with the relevant loss curves. Along with the models, this repo contains code for training the models yourself. Why would you want to do this, since I'm brilliant and my best model has a stunningly high accuracy rate of 0% (not a joke)? Search me, but you can!

This code was designed with the decomposition of multi-pulse radio transient profiels in mind. It works just about as well as `scipy.signal.find_peaks`, but you should probably just use <a href = "https://iopscience.iop.org/article/10.3847/1538-4357/ad1ce7/meta">CLEAN</a>.
