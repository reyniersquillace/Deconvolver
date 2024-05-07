import numpy as np

def main():

    mass = np.loadtxt("output/mass.txt")

    mass_stretched = np.zeros(3000)

    counter = 0
    for i in range(200):
        el = [mass[i]]*15
        mass_stretched[counter:counter+15] = el
        counter += 15

    np.savetxt('output/mass_stretched_15.txt', mass_stretched)

    return

if __name__=='__main__':
    main()
