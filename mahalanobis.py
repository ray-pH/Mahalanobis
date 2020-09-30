import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def getcol(mat, col):
	return np.array([mat.transpose()[col]]).transpose()

def getminormajor(x,y,chi):
	cov = np.cov(x, y)
	mean_x = np.mean(x)
	mean_y = np.mean(y)
	mu = np.array([[mean_x],[mean_y]])

	w,v = np.linalg.eig(cov)
	minor = np.sqrt(chi*w[0])*getcol(v,0)
	min1 = (mu + minor).transpose()[0]
	min2 = (mu - minor).transpose()[0]
	major = np.sqrt(chi*w[1])*getcol(v,1)
	maj1 = (mu + major).transpose()[0]
	maj2 = (mu - major).transpose()[0]

	thex = [mu[0], min1[0], min2[0], maj1[0], maj2[0]]
	they = [mu[1], min1[1], min2[1], maj1[1], maj2[1]]
	return [thex,they]
	


def confidence_ellipse(x, y, ax, chi, facecolor='none', **kwargs):
    """
    Parameters
    ----------
    x, y : array-like, shape (n, ) Input data.
    ax : matplotlib.axes.Axes The axes object to draw the ellipse into.
    n_std : float The number of standard deviations to determine the ellipse's radiuses.
    **kwargs Forwarded to `~matplotlib.patches.Ellipse`
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    n_std = np.sqrt(chi)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def main():
	fig, ax = plt.subplots()

	datax = [108.28, 152.36, 95.04, 65.45, 62.97, 263.99, 265.19, 285.06, 92.01, 165.68]
	datay = [17.05, 16.59, 10.91, 14.14, 9.52, 25.33, 18.54, 15.73, 8.1, 11.13]
	alpha = 0.05
	showcoords = True
	ax.set_xlim([-120, 400])
	ax.set_ylim([0, 30])

	ax.scatter(datax, datay)
	chi = chi2.ppf(1-alpha,2)
	minmaj = getminormajor(datax,datay,chi)
	ax.scatter(minmaj[0],minmaj[1])

	if showcoords:
		for i in range(len(minmaj[0])):
			xs,ys = minmaj
			text = "({:.2f}, {:.2f})".format(float(xs[i]),float(ys[i]))
			ax.text(xs[i],ys[i],text)
	confidence_ellipse(datax, datay, ax, chi, edgecolor='red')
	plt.show()
	
if __name__ == '__main__':
	main()