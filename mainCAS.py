import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from astropy.visualization import LogStretch
from astropy.modeling import models
import photutils
import time
import statmorph
import argparse
import os,sys
# %matplotlib inline


def normalize(image):
        m, M = np.min(image), np.max(image)
        return (image-m) / (M-m)


def morpho(myFile, gain):

    global args

    ##### Load the image
    image = np.loadtxt(myFile) 
    image = np.float64(image)
    
    ##### ??
    log_stretch = LogStretch(a=10000.0)

    ##### Normalize the image
    image = normalize(image)

    ##### Compute and plot the segmentation map
    if args.plot:
        plt.figure()
        left = plt.subplot(121)
        right = plt.subplot(122)

    threshold = photutils.detect_threshold(image, 1.5)
    npixels = 5  # minimum number of connected pixels
    segm = photutils.detect_sources(image, threshold, npixels)
    # keep only the largest segment
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    if args.plot:
        left.imshow(segmap, origin='lower', cmap='gray')

    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5
    if args.plot:
        right.imshow(segmap, origin='lower', cmap='gray')
        plt.show()

    ##### Compute the morphology using the gain and the segmap
    start = time.time()
    source_morphs = statmorph.source_morphology(image, segmap, gain=gain)
    print('Time: %g s.' % (time.time() - start))

    morph = source_morphs[0]

    if args.save:
        file = open(myFile + ".txt", "w") 
        file.write('xc_centroid ='+ str(morph.xc_centroid)+"\n")
        file.write('yc_centroid ='+ str(morph.yc_centroid)+"\n")
        file.write('ellipticity_centroid ='+ str(morph.ellipticity_centroid)+"\n")
        file.write('elongation_centroid ='+ str(morph.elongation_centroid)+"\n")
        file.write('orientation_centroid ='+ str(morph.orientation_centroid)+"\n")
        file.write('xc_asymmetry ='+ str(morph.xc_asymmetry)+"\n")
        file.write('yc_asymmetry ='+ str(morph.yc_asymmetry)+"\n")
        file.write('ellipticity_asymmetry ='+ str(morph.ellipticity_asymmetry)+"\n")
        file.write('elongation_asymmetry ='+ str(morph.elongation_asymmetry)+"\n")
        file.write('orientation_asymmetry ='+ str(morph.orientation_asymmetry)+"\n")
        file.write('rpetro_circ ='+ str(morph.rpetro_circ)+"\n")
        file.write('rpetro_ellip ='+ str(morph.rpetro_ellip)+"\n")
        file.write('rhalf_circ ='+ str(morph.rhalf_circ)+"\n")
        file.write('rhalf_ellip ='+ str(morph.rhalf_ellip)+"\n")
        file.write('r20 ='+ str(morph.r20)+"\n")
        file.write('r80 ='+ str(morph.r80)+"\n")
        file.write('Gini ='+ str(morph.gini)+"\n")
        file.write('M20 ='+ str(morph.m20)+"\n")
        file.write('F(G, M20) =' + str(morph.gini_m20_bulge)+"\n")
        file.write('S(G, M20) =' + str(morph.gini_m20_merger)+"\n")
        file.write('sn_per_pixel ='+ str(morph.sn_per_pixel)+"\n")
        file.write('C ='+ str(morph.concentration)+"\n")
        file.write('A ='+ str(morph.asymmetry)+"\n")
        file.write('S ='+ str(morph.smoothness)+"\n")
        file.write('sersic_amplitude ='+ str(morph.sersic_amplitude)+"\n")
        file.write('sersic_rhalf ='+ str(morph.sersic_rhalf)+"\n")
        file.write('sersic_n ='+ str(morph.sersic_n)+"\n")
        file.write('sersic_xc ='+ str(morph.sersic_xc)+"\n")
        file.write('sersic_yc ='+ str(morph.sersic_yc)+"\n")
        file.write('sersic_ellip ='+ str(morph.sersic_ellip)+"\n")
        file.write('sersic_theta ='+ str(morph.sersic_theta)+"\n")
        file.write('sky_mean ='+ str(morph.sky_mean)+"\n")
        file.write('sky_median ='+ str(morph.sky_median)+"\n")
        file.write('sky_sigma ='+ str(morph.sky_sigma)+"\n")
        file.write('flag ='+ str(morph.flag)+"\n")
        file.write('flag_sersic ='+ str(morph.flag_sersic)+"\n")
        file.close() 
    else:
        print('xc_centroid =', morph.xc_centroid)
        print('yc_centroid =', morph.yc_centroid)
        print('ellipticity_centroid =', morph.ellipticity_centroid)
        print('elongation_centroid =', morph.elongation_centroid)
        print('orientation_centroid =', morph.orientation_centroid)
        print('xc_asymmetry =', morph.xc_asymmetry)
        print('yc_asymmetry =', morph.yc_asymmetry)
        print('ellipticity_asymmetry =', morph.ellipticity_asymmetry)
        print('elongation_asymmetry =', morph.elongation_asymmetry)
        print('orientation_asymmetry =', morph.orientation_asymmetry)
        print('rpetro_circ =', morph.rpetro_circ)
        print('rpetro_ellip =', morph.rpetro_ellip)
        print('rhalf_circ =', morph.rhalf_circ)
        print('rhalf_ellip =', morph.rhalf_ellip)
        print('r20 =', morph.r20)
        print('r80 =', morph.r80)
        print('Gini =', morph.gini)
        print('M20 =', morph.m20)
        print('F(G, M20) =', morph.gini_m20_bulge)
        print('S(G, M20) =', morph.gini_m20_merger)
        print('sn_per_pixel =', morph.sn_per_pixel)
        print('C =', morph.concentration)
        print('A =', morph.asymmetry)
        print('S =', morph.smoothness)
        print('sersic_amplitude =', morph.sersic_amplitude)
        print('sersic_rhalf =', morph.sersic_rhalf)
        print('sersic_n =', morph.sersic_n)
        print('sersic_xc =', morph.sersic_xc)
        print('sersic_yc =', morph.sersic_yc)
        print('sersic_ellip =', morph.sersic_ellip)
        print('sersic_theta =', morph.sersic_theta)
        print('sky_mean =', morph.sky_mean)
        print('sky_median =', morph.sky_median)
        print('sky_sigma =', morph.sky_sigma)
        print('flag =', morph.flag)
        print('flag_sersic =', morph.flag_sersic)


parser = ""
args = ""

def init_args():

    global parser, args
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="the file containing the image to analyse")
    parser.add_argument("-g", "--gain", type=int, default=1000.0, help="gain value, necessary to compute the morph statistics.")
    parser.add_argument("-s", "--save", action="store_true", help="save results in a txt file instead of printing it")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the segmentation map")
    args = parser.parse_args()

if __name__ == "__main__":
    init_args()
    morpho(args.file, args.gain)
