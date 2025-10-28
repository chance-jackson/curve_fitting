import numpy as np
import matplotlib.pyplot as plt
import h5py
import optimize

f = h5py.File('./data.hdf', 'r')

print(f.keys())

print(f['data'].keys())

xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

plt.scatter(xpos, ypos)
plt.show()

def fitting_func(x,a,b,c,d,e,f,g):
    return g * x ** 6 + f * x ** 5 + e * x ** 4 + a * x ** 3 + b * x ** 2 + c * x + d
N_params = 7
def cost_func(fit, data, guess_params):
    resid = fit(data[0], *guess_params) - data[1] #find resid
    square_resid = np.power(resid,2) #square em

    return np.sum(square_resid) #return their sum
#print(cost_func(fitting_func, (xpos, ypos), np.ones(N_params))) #appears to function
#We need to minimize this cost function, so we can use Newton's method
minima, x_hist, y_hist, n_iter = optimize.newtons(lambda guess_params: cost_func(fitting_func, (xpos, ypos), guess_params), np.ones(N_params), rate = 1)

print("The minima : ",minima, f"was found in {n_iter} iterations")
#print(cost_func(fitting_func, (xpos, ypos), minima))
#print(n_iter)

######## CALCULATION OF R2 ####################
dof_res = len(xpos) - N_params - 1
dof_tot = len(xpos) - 1
ss_tot = np.sum(np.square(ypos - np.mean(ypos)))/dof_res
ss_res = np.sum(np.square(ypos - fitting_func(xpos, *minima)))/dof_tot

adjusted_R2 = 1 - ss_res/ss_tot
print(r"Adjusted R2: ", adjusted_R2)

fig,axs = plt.subplots(nrows = 2, figsize = (8,6), sharex = True)
axs[0].scatter(xpos, ypos, label="Data")
axs[0].set_title("Fitting 6th Order Polynomial")
axs[0].set_ylabel("Y")
x_plot = np.linspace(min(xpos), max(xpos), 10000)
axs[0].plot(x_plot, fitting_func(x_plot, *minima), color = 'red', label = "Best-Fit")
axs[1].scatter(xpos, fitting_func(xpos, *minima) - ypos)
axs[1].set_ylabel("Residuals")
axs[1].set_xlabel("X")
axs[0].legend(loc = "lower right")
plt.savefig("6thorderpoly.png", dpi = 300)
