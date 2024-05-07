from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from matplotlib import rc
from matplotlib import pyplot as plt
import os,sys

def get_names(run_string):

    all_grids = run_string.strip().split(",")

    path='/home/j.rose/Projects/CAMELS/GRIDS/'

    names = []
    all_runs = []
    for run in all_grids:

        sims = os.listdir(f'{path}/{run}/')
        sims = [el for el in sims if 'WDM' in el[:3]]
        nums = [int(el.split("M")[1]) for el in sims if 'WDM' in el[:3]]
        sort_sims = [x for y,x in sorted(zip(nums, sims))]
        for sim in sort_sims:

            if 'CDM' in sim:
                all_runs.append(f'{path}/{run}/RUNs/{sim}/')
                names.append('CDM')
            elif 'WDM' in sim:
                names.append(sim)
                all_runs.append(f'{path}/{run}/{sim}/')

    return names, all_runs

def get_snap(snap_str):
    
    snap_li_str = snap_str.strip().split(",")
    snap_li = [int(el) for el  in snap_li_str]

    if len(snap_li) == 1:
        snap_li = snap_li[0]

    return snap_li

def set_ticks(ax):
    ax.tick_params('both', which='minor', length=4, direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params('both', which='major', length=8, direction='in', bottom=True, top=True, left=True, right=True)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    return 

def set_plot_params(nrows=1, ncols=1, figsize=None):

    plt.rc('font',**{'family':'STIXGeneral'})
    plt.rc('text', usetex=True)

    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)

    if figsize is None:
        fig, ax = plt.subplots(nrows,ncols, figsize=(ncols*5 + (ncols)*2,nrows*5+(nrows-1)*2))
    else:
        fig, ax = plt.subplots(nrows,ncols, figsize=figsize)

    if type(ax)==type(np.zeros(1)):
        for a in ax.ravel():
            set_ticks(a)
    else:
        set_ticks(ax)

    return fig, ax


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class ObsData():

    def __init__(self, path, points=False, lines=False, error=False, y_is_log=True, xerror=False, yerror=False):
        self.path = path
        self.y_is_log = y_is_log
        self.error = error
        self.xerror = xerror
        self.yerror = yerror

        self.cols = None
        self.data = self.read_data()
        
        if xerror is not False or yerror is not False:
            self.error = xerror if xerror is not False else yerror 

        if lines:
            self.lines = self.calc_lines()
        if points:
            self.xpoints, self.ypoints = self.get_points()
        if self.error is not False:
            self.xpoints, self.ypoints, self.xerr, self.yerr = self.calc_error()

    def calc_error(self):
        xpoints = []
        ypoints = []
        xerr = []
        yerr = []
        for col in self.data:
            x = np.array(self.data[col]['x'])
            y = np.array(self.data[col]['y'])

            ymax = np.max(y)
            ymin = np.min(y)
            xmax = np.max(x)
            xmin = np.min(x)

            xpoint = np.median(x[(x>xmin) & (x<xmax)])
            ypoint = np.median(y[(y>ymin) & (y<ymax)])

            if len(x)<5:
                if self.xerror is not False:
                    ypoint = np.median(y)
                elif self.yerror is not False:
                    xpoint = np.median(x)
                else:
                    bounds = np.array([xmax, xmin, ymax, ymin])
                    cut = (bounds-xpoint == 0) | (bounds-ypoint == 0)
                    if cut[0]:
                        xmax = xpoint + (xpoint - xmin)
                    if cut[1]:
                        xmin = xpoint - (xmax - xpoint)
                    if cut[2]:
                        ymax = ypoint + (ypoint - ymin)
                    if cut[1]:
                        ymin = ypoint - (ymax - ypoint)                   

            xpoints.append(xpoint)
            ypoints.append(ypoint)
            if self.error == 'same':
                xerr.append((xmax-xpoint) + (xpoint-xmin) /2)
                yerr.append((ymax-ypoint) + (ypoint-ymin) /2)
            elif self.error == 'unique':
                xerr.append([xmax-xpoint, xpoint-xmin])
                yerr.append([ymax-ypoint, ypoint-ymin])
            else:
                raise KeyError(f"{self._error} not supported for error type, please use 'unique' or 'same")
        return xpoints, ypoints, xerr, yerr

    def get_points(self):
        xpoints = [self.data[key]['x'] for key in self.data]
        ypoints = [self.data[key]['y'] for key in self.data]
        return xpoints, ypoints

    def lin(self, x, m, b):
        return m*x + b

    def calc_lines(self):
        lines = {el:[] for el in self.cols}
        for col in self.cols:
            x = self.data[col]['x']
            y = self.data[col]['y']
            guess = [(y.iloc[-1]-y.iloc[0]) / (x.iloc[-1]-x.iloc[0]), 1]
            if self.y_is_log:
                y = np.log10(y)
            popt, pcov = curve_fit(self.lin, x, y, p0=guess) 
            lines[col] = popt
        return lines

    def read_data(self):
      
        df = pd.read_csv(self.path)
        self.cols = [el for el in set(df.columns) if 'named' not in el]

        x = None
        y = None
        col_name = ''
        data = dict()
        for col in df.columns:
            if x is None:
                x = df.loc[1:,col].dropna().astype(float)
                col_name = col
            elif y is None:
                y = df.loc[1:,col].dropna().astype(float)
                data[col_name] = {'x':x, 'y':y}
                x = None
                y = None
                col_name = ''

        return data


                
