import pydisort

def disort_config(disort, nstr, nlyr, ncol, nwave):
    disort.nwave(nwave)
    disort.ncol(ncol)
    disort.ds().nlyr = nlyr
    disort.ds().nstr = nstr
    disort.ds().nmom = nstr
    disort.ds().nphase = nstr
