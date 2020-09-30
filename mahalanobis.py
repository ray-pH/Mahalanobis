import numpy as np
from scipy.stats import chi2
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def getcol(mat, col):
	return np.array([mat.transpose()[col]]).transpose()

class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

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
	
def getoutlier(x,y,chi):
	out_x,out_y = [],[]
	cov = np.cov(x,y)
	icov = np.linalg.inv(cov)
	mu = [np.mean(x), np.mean(y)]
	for i in range(len(x)):
		vec = [x[i],y[i]]
		dist = distance.mahalanobis(vec,mu,icov)
		if(dist**2 > chi): 
			out_x.append(x[i])
			out_y.append(y[i])
	return [out_x,out_y]



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

	# datax = [108.28, 152.36, 95.04, 65.45, 62.97, 263.99, 265.19, 285.06, 92.01, 165.68]
	# datay = [17.05, 16.59, 10.91, 14.14, 9.52, 25.33, 18.54, 15.73, 8.1, 11.13]
	datax = [15.150,14.680,13.097,13.695,16.065,13.632,13.219,12.830,14.259,13.825,13.338,12.696,13.770,13.230]
	datay = [11.221,11.723,12.280,12.252,10.343,12.273,11.560,11.971,10.546,10.857,11.848,11.557,11.499,12.342]
	alpha = 0.1
	# alpha = 0.25
	coord_fontsize = 7
	show_coords = True
	show_line = True
	highlight_outlier = True
	ax.set_xlim([11, 17])
	ax.set_ylim([8.5, 14.5])

	#colors
	outlier_color = 'green'
	ellipse_color = 'red'
	data_color = '#4293f5'
	minmaj_color = 'orange'
	line_color = 'blue'

	print(np.cov(datax,datay))

	chi = chi2.ppf(1-alpha,2)
	minmaj = getminormajor(datax,datay,chi)

	if show_line:
		ax.plot(minmaj[0][1:3],minmaj[1][1:3], linestyle='dashed', color=line_color, alpha=0.3, zorder=0)
		ax.plot(minmaj[0][3:5],minmaj[1][3:5], linestyle='dashed', color=line_color, alpha=0.3, zorder=0)
	ax.scatter(datax, datay, color=data_color)
	ax.scatter(minmaj[0],minmaj[1], color=minmaj_color)

	if show_coords:
		for i in range(len(minmaj[0])):
			xs,ys = minmaj
			text = "({:.2f}, {:.2f})".format(float(xs[i]),float(ys[i]))
			ax.text(xs[i],ys[i],text, fontsize=coord_fontsize)

	if highlight_outlier:
		outs = getoutlier(datax,datay,chi)
		ax.scatter(outs[0],outs[1], color=outlier_color)
		print("jumlah outlier :",len(outs[0]))


	confidence_ellipse(datax, datay, ax, chi, edgecolor=ellipse_color)

	zp = ZoomPan()
	scale = 1.1
	figZoom = zp.zoom_factory(ax, base_scale = scale)
	figPan = zp.pan_factory(ax)
	plt.show()
	
if __name__ == '__main__':
	main()