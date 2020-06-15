import argparse, sys, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

lib_path = os.path.abspath(os.path.join(__file__, '..', "..", 'libs'))
sys.path.append(lib_path)

from pic_process import *
from minkos import *

parser = ""
args = ""


def get_right(F, U, chi, arg):
    """
        Retourne la fonctionelle correspondant à l'argument et sa couleur
    """
    x = 0
    if arg == "f":
        x = F
    elif arg == "u":
        x = U
    elif arg == "chi":
        x = chi
    return x, func_col(arg)


def main(myFile):
    max_lin = args.max
    size_window = [10, 8]
    fig = plt.figure(figsize=(*size_window,))

    beta = fig.add_subplot(121)
    delta = fig.add_subplot(122)

    x = np.linspace(0.0, max_lin, 100)

    #file1, name, ext = get_image(myFile)
    file1, name = charger_fichier_A(myFile)
    print("Processing", name, "...")

    if args.smooth:
        file1 = smooth_file(file1, args.smooth)

    if args.contrastLinear:
        file1 = contrastLinear(file1, args.contrastLinear)

    F, U, Chi = calcul_fonctionelles(file1, max_lin)

    h, col = get_right(F, U, Chi, args.functional)
    h = h / coef_normalization_functional(h)

    list_max_min = []

    for i, v in enumerate(h):
        if i > 1:
            if h[i - 2] > h[i - 1] and h[i - 1] < h[i]:
                print(x[i - 1])
                list_max_min += [x[i - 1]]
            if h[i - 2] < h[i - 1] and h[i - 1] > h[i]:
                print(x[i - 1])
                list_max_min += [x[i - 1]]

    print(list_max_min)

    axfreq = plt.axes([0.563, 0.04, 0.324, 0.03], facecolor="white")
    sthresh = Slider(axfreq, 'Freq', 0, x[-1], valinit=10, valstep=3)

    #beta.plot(x,h/np.max(h))
    #beta.plot(xf,ff, linewidth = 4)
    if False:
        for i in ["f", "u","chi"]:
            h,col = get_right(F,U,Chi, i)
            h = h/coef_normalization_functional(h)
            delta.plot(x,h)
        delta.set_title(args.functional)
        delta.legend(["F","U","Chi"])
    #delta.set_xlabel("Threshold")
    # beta.plot(x,h/np.max(h))
    # beta.plot(xf,ff, linewidth = 4)
    delta.plot(x, h)
    delta.set_title(args.functional)
    # delta.set_xlabel("Threshold")

    show_threshold(beta, file1, 10)

    def update(val):
        print(1)
        threshold = sthresh.val
        show_threshold(beta, file1, threshold)
        # beta.canvas.draw_idle()
        fig.canvas.draw()

    sthresh.on_changed(update)

    # plt.tight_layout()

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.inaxes == delta:
            sthresh.val = event.xdata
            update(True)
            axfreq.clear()
            sthresh.__init__(axfreq, 'Freq', 0, x[-1], valinit=round(event.xdata), valstep=3)

    beta.plot()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


def show_threshold(ax, img, threshold):
    ax.cla()
    michel = supra_boucle(img, threshold)
    ax.set_title("Threshold - " + str(round(threshold)))
    ax.imshow(michel, cmap="viridis")


def supra_boucle(file1, threshold):
    file1 = np.copy(file1)
    for i in range(len(file1)):
        for j in range(len(file1[i])):
            pix = file1[i][j]
            if pix > threshold:
                file1[i][j] = 1
            else:
                file1[i][j] = 0
    return file1


def init_args():
    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = float, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int, default=40)
    parser.add_argument("-dat", "--dat", action="store_true", help="file is in dat format")
    parser.add_argument("-smooth", "--smooth", type=int, help="smooth", default=0)
    parser.add_argument("-n", "--name", type=str, help="name of file")
    parser.add_argument("-f", "--functional", type=str, help="name of functional to show", choices=['f', 'u', 'chi'],
                        default="f")
    parser.add_argument("-nonorm", "--nonorm", action="store_true", help="No normalisation")
    parser.add_argument("-maxfiles", "--maxfiles", type=int, help="max_nb of files to process, 0 for infnity",
                        default=3)
    args = parser.parse_args()

    args.drawall = True


if __name__ == "__main__":
    init_args()
    main(args.file)