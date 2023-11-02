import matplotlib.pyplot as plt
import numpy as np

def plt_sinkhorn_error(errs,P):
    plt.figure(figsize=(9,5))

    plt.subplot(2,1,1)
    plt.plot([np.log(x[0]) for x in errs])
    plt.title('Row sum constraint violation (log scale)')
    plt.ylabel("$\log \ \ ||P 1 - a||_1$")
    plt.xlabel('Sinkhorn iteration')


    plt.subplot(2,1,2)
    plt.plot([np.log(x[1]) for x in errs])
    plt.title('Column sum constraint violation (log scale)')
    plt.ylabel("$\log \ \ ||P^T 1 - b||_1$")
    plt.xlabel('Sinkhorn iteration')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.imshow(P,origin='lower',cmap='inferno')
    plt.axis('off')
    plt.show()

    ## to do -- refactor for marginals like in phd histo plots :)

    return