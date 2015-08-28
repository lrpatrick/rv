"""
Author: LRP
Date: 18-08-2015
Description:
Calculate radial velocity for RSGs based on a cross-correlateion onto rest
wavelength by comparison with atmospheric profile following (Lapenna et al.
2015).
Radial velocity is then calculated by a comparison with a synthetic RSG
spectrum

Usage:
import rv

rv.rv(sci, at_spec, fakersgspec)

"""
from __future__ import print_function

import astropy.io.fits as pyfits
import glob
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.io.idl import readsav

# import AR
import CrossCorrelate as cc
from degrade import degrader
from kmostcorr import rescale


C = 3.0*10**5  # km/s


def lamaxis(lmin, step, size):
    """Wrap np.linspace to create x-axis for spectral data"""
    return np.linspace(lmin, lmin + (step*(size - 1)), size)

# Implicit Variables available to all functions:
# Atmospheric spectrum:
atmos = pyfits.open('/home/lee/Work/python/lib/atmos_S_J.fits')
atspec_hres = atmos[0].data
xat_hres = lamaxis(atmos[0].header['CRVAL1'], atmos[0].header['CDELT1'],
                   np.shape(atspec_hres)[0])*10**-4

# Fake spectra:
allgrid = readsav('/home/lee/Work/RSG-JAnal/models/MODELSPEC_2013sep12_nLTE_R10000_J_turb_abun_grav_temp-int.sav')
# fspec = allgrid['modelspec'][0][0][0, 0, 0, 6]
fspec = allgrid['modelspec'][0][0][13, 4, 5, 6]
fx = allgrid['modelspec'][0][2]


lines = np.genfromtxt('/home/lee/Work/python/lib/lines.txt')[:, 1]
lines = np.delete(lines, (3, 7))


class SciComb(object):
    """Class for combined science cube"""
    def __init__(self, sfile):
        self.sfile = sfile
        self.date = sfile[0].header['DATE-OBS'][0:10]
        self.cube = np.nan_to_num(sfile[1].data)
        self.bx, self.by = brighest_pix(self.cube)
        self.bspec = self.cube[:, self.bx, self.by] \
            / np.median(self.cube[:, self.bx, self.by])
        # self.bspec[self.bspec == 0.0] = 1.0
        self.ifu = np.int(sfile[1].header['*ARM*'].cards[0][0][11:13])
        self.name = sfile[1].header['EXTNAME'][:-5]
        self.lmin = sfile[1].header['CRVAL3']
        self.delt = sfile[1].header['CDELT3']
        self.x = np.linspace(self.lmin, self.lmin +
                             (self.delt*(np.shape(self.bspec)[0] - 1)),
                             np.shape(self.bspec)[0])
        self.roi = np.where((self.x > 1.17) & (self.x < 1.22))


class TellCube(object):
    """Class for Telluric data cube"""
    def __init__(self, tfile):
        self.tfile = tfile
        self.tcube = pyfits.open(tfile)
        self.date = self.tcube[0].header['DATE-OBS'][0:10]
        self.data = self.tcube[1::2]


class TellCor(object):
    """Telluric correct spectra which have been preped"""
    def __init__(self, s1, s2, x):
        self.sci = s1 / np.median(s1)
        self.rawt = s2
        self.x = x
        self.tcc, self.shift = cc.ccshift(s1, self.rawt, self.x, quiet=False)
        self.tsr, self.c = rescale(s1, self.tcc)
        self.tcor = self.sci / self.tsr


def wraprv(scipath, ftell):
    """
        Wrap the rv.rv routine by giving it a list of spectra

        Usage:

        rv.wraprv('path/to/sci_combined/files', '/path/telluric_YJYJYJ.fits')
    """
    hcorr = raw_input('[INFO] Please enter heliocentric corr. for target:\n')
    hcorr = float(hcorr)
    # scipath = '../KMOSreduction/spark-1.3.3/KMOSscience/'
    # ftell = '../telluric_correction/tellurics/kmo-1.3.0/telluric_YJYJYJ_3072.fits'
    scifiles = sorted(glob.glob(scipath + '*sci_combined_N*'))

    # Avoid repeating this procedure by assuming all spectra have same x-axis
    tmpsci = SciComb(pyfits.open(scifiles[0]))
    print('[INFO] Match sampling and degrade atmos. spec. to match observed')
    atssam = resam(xat_hres, atspec_hres, tmpsci.x)
    # Clean up edges -- taken from N6822_Vrad.v2.py -- is this necesssary?
    atssam[np.where(atssam < 0.0)[0]] = 0
    print('[INFO] Degrade atmos spec to match science:')
    # From ESO's ISAAC decommissioned webpages R = 40000
    atspec = degrader(tmpsci.x, atssam, 40000, 3000)

    rvall = []
    rvstd = []
    name = []
    tcspec = []
    strest = []

    for scifile in scifiles:
        print('[INFO] Use rv.rv for ', scifile)
        namei, tcspeci, rvi, erri, sr = rv(scifile, ftell, atspec, hcorr)
        rvall = np.append(rvall, rvi)
        rvstd = np.append(rvstd, erri)
        name = np.append(name, namei)
        tcspec = np.append(tcspec, tcspeci)
        strest = np.append(strest, sr)
    print('--'*10)
    print('[INFO] Average RV for the sample: {} +/- {}'.format(rvall.mean(),
                                                               rvall.std()))
    print('--'*10)
    return name, tcspec, rvall, rvstd, strest


def writervfile(fname, head, names, rvs, errs):
    out = np.column_stack((names, rvs, errs))
    np.savetxt(fname, out, header=head, fmt='%s')


def rv(fsci, ftell, atspec, hcorr):
    """Calculate RV for a science spectrum extracted from brighest pixel"""

    scic = SciComb(pyfits.open(fsci))
    # print('[INFO] Match sampling and degrade atmos. spec. to match observed')
    # atssam = resam(xat_hres, atspec_hres, scic.x)
    # # Clean up edges -- taken from N6822_Vrad.v2.py -- is this necesssary?
    # atssam[np.where(atssam < 0.0)[0]] = 0
    # print('[INFO] Degrade atmos spec to match science:')
    # # From ESO's ISAAC decommissioned webpages R = 40000
    # atspec = degrader(scic.x, atssam, 40000, 3000)
    # Degrade then resample:
    # atdeg = degrader(xat_hres, atspec_hres, 40000, 3000)
    # atspec = resam(xat_hres, atdeg, scic.x)

    # For cross-correlation to rest use telluric index:
    it = np.where((scic.x > 1.12) & (scic.x < 1.15))[0]
    xtell = scic.x[it]
    scixtell = scic.bspec[it]
    atxtell = atspec[it]
    scixtell_, sr = cc.ccshift(atxtell, scixtell, xtell, quiet=False)

    # Implement shift-to-rest (sr) in diagnostic region
    xdiag = scic.x[scic.roi]
    scidiag = scic.bspec[scic.roi]
    atdiag = atspec[scic.roi]
    print('[INFO] Shift science spectrum to rest using telluric features:')
    scirest, s2 = cc.ccshift(atdiag, scidiag, xdiag, shift1=sr, quiet=False)

    # Telluric correct:
    print('[INFO] Telluric correct data once shifted onto rest wavelength:')
    tellcube = TellCube(ftell)
    tspec = tellcube.data[scic.ifu - 1].data
    scitc = TellCor(scirest, tspec[scic.roi], scic.x[scic.roi])

    # Prepare fake spectrum:
    # Match sampling between fake spectrum and observations
    fxtrim, ftrim = trimspec(scic.x, fx, fspec)
    # Degrade & resample:
    fsam = resam(fx, fspec, scic.x)

    # Degrade atmos spec to match science:
    # fdeg = degrader(scic.x, fsam, 10000, 10000)
    fdiag = fsam[scic.roi]
    print('[INFO] Calculate RV shift:')
    scirv, rvs = cc.ccshift(scitc.tcor, fdiag, xdiag, quiet=False)
    rvkms = lambda shift, delta, lam: shift*delta*C / lam
    rvi = rvkms(rvs, scic.delt, 1.2)

    # Calculate errors:
    slbl = linebyline(xdiag, scitc.tcor, fdiag)
    slbl = slbl[np.where(slbl != 0.0)]
    rvlbl = rvkms(slbl, scic.delt, 1.2)
    avrv = rvcorr(np.mean(rvlbl), hcorr)
    erv = np.std(rvlbl) / np.sqrt(np.shape(rvlbl)[0])
    print('[INFO] Average RV line by line: ', avrv, '+/-', erv)
    print('[INFO] RV derived using the whole region: ', rvcorr(rvi, hcorr))
    return scic, scitc, avrv, erv, sr

# rvb = np.zeros((np.shape(errall)[0], 2))
# rvb[:, 1][3:] = errall[:-3]
# rvb[:, 0][3:] = rvall[:-3]
# rvb[:, 0][0:3] = rvall[-3:]
# rvb[:, 1][0:3] = errall[-3:]
# plt.errorbar(np.arange(0, np.shape(rvpub)[0]), rvpub[:, 0],
#              yerr=rvpub[:, 1], fmt='o', color='blue')
# plt.errorbar(np.arange(0, np.shape(rvpub)[0]), sortedrvs[:, 0],
#              yerr=sortedrvs[:, 1], fmt='o', color='green')
# plt.errorbar(np.arange(0, np.shape(rvpub)[0]), rvsnew[:, 0],
#              yerr=rvsnew[:, 1], fmt='o', color='red')


def rvcorr(rv, hcorr):
    """
        Correct the radial velocity
            1. Heliocentric correction from ESO's airmass calculator
            2. Difference between vacuum and air
    """
    # hcorr = 7.75
    airvac = 82.22
    return rv + hcorr - airvac


def defidx(w1):
    """Define regions with strong lines to compute cross-correlation"""
    idx = [np.where((w1 > 1.1760) & (w1 < 1.1805))[0]]
    idx.append(np.where((w1 > 1.1820) & (w1 < 1.18495))[0])
    idx.append(np.where((w1 > 1.1869) & (w1 < 1.1899))[0])
    idx.append(np.where((w1 > 1.1965) & (w1 < 1.19986))[0])
    idx.append(np.where((w1 > 1.20133) & (w1 < 1.205))[0])
    idx.append(np.where((w1 > 1.20691) & (w1 < 1.2124))[0])
    return idx


def linebyline(x, tcspec, fspec):
    """
        Calculate shift to rest wavelength for each line individually
    """
    lines = defidx(x)
    # wid = 0.0010
    slbl = []
    for l in lines:
        # lidx = np.where((x > l - wid) & (x < l + wid))[0]
        # Generate fake spectrum with one line surrounded by ones:
        # fones = onespec(x, fspec[lidx], lidx)
        slbli, tmp = cc.crossc(fspec, tcspec,
                               i1=l[0], i2=l[-1])

        slbl = np.append(slbl, slbli*-1)
    return slbl


def onespec(x, line, lidx):
    """
        Create a spectrum of ones with a spectral line imprinted

        x : numpy.ndarray
            x-axis
        line : numpy.ndarray
            line to put into an array of ones
        lidx : numpy.ndarray
            Position to place the line (line index)
        Return:
        fline : numpy.ndarray
            Spectrum containing real line in a sea of ones ...
    """
    fline = np.ones(x.shape)
    fline[lidx] = line
    return fline


def brighest_pix(data):
    """
        Define the brightest pixel from a 3D data cube the median along the
        spectral axis

        Arguments:
        data : numpy.ndarray
            3D data cube from a reconstructed KMOS IFU

        Output:
        x, y : int
            index of the brightest pixel within the array
    """
    med_spec = np.median(data, axis=0)
    x, y = np.unravel_index(med_spec.argmax(), med_spec.shape)
    return x, y


def resam(x, y, xnew):
    s = UnivariateSpline(x, y, s=1)
    return s(xnew)


def trimspec(w1, w2, s2):
    """Trim s2 and w2 to match w1"""
    roi = np.where((w2 > w1.min()) & (w2 < w1.max()))[0]
    return w2[roi], s2[roi]
