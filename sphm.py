import numpy as np
from scipy.special import sph_harm


def sh_all(theta,phi,Lmaxmax):
    
    Ylm_r = np.zeros((Lmaxmax+1, 2*Lmaxmax+1, len(theta)))
    Ylm_i = np.zeros((Lmaxmax+1, 2*Lmaxmax+1, len(theta)))
    Ylm = np.zeros((Lmaxmax+1, 2*Lmaxmax+1, len(theta)))

    for l in range(0,Lmaxmax+1):
      for m in range(-l, l+1):
        Ylm_r[l,m+l,:] = getattr(sph_harm(m,l,theta,phi), 'real')
        Ylm_i[l,m+l,:] = getattr(sph_harm(m,l,theta,phi), 'imag')

      for m in range(-l, l+1):
        Ylm[l][m+l] = Ylm_r[l,m+l,:]
        if(m>0):
          Ylm[l][m+l] = Ylm_r[l][m+l] * np.sqrt(2)
        elif(m<0):
          Ylm[l][l+m] = np.power(-1.0,m) * Ylm_i[l][l+m] * np.sqrt(2)

    return Ylm_r,Ylm_i, Ylm

def sh_decomposition(f,fi,Ylm_r,Ylm_i,dA,surfaceArea,Lmax):
    flm_r = []
    flm_i = []
    flm = []
    
    for l in range(0,Lmax+1):
      flm_r += [[]]
      flm_i += [[]]
      flm += [[]]

      for m in range(-l, l+1):
        flm_r[l] += [4*np.pi * np.sum(dA[:,] * (f[:,] * Ylm_r[l,m+l,:] + fi[:,] * Ylm_i[l,m+l,:]))/(surfaceArea)]
        flm_i[l] += [4*np.pi * np.sum(dA[:,] * (fi[:,] * Ylm_r[l,m+l,:] - f[:,] * Ylm_i[l,m+l,:]))/(surfaceArea)]

      #print(flm_r[l])
      for m in range(-l, l+1):
        flm[l] += [flm_r[l][l+m]]
        if(m>0):
          flm[l][m+l] = flm_r[l][m+l]*np.sqrt(2)
        elif(m<0):
          flm[l][m+l] = flm_i[l][-m+l]*np.sqrt(2)

    return flm_r, flm_i, flm

def sh_energy_representation(flm, Ylm, Lmax, dA):
    Sl = np.zeros(Lmax+1)
    Slm = np.zeros((Ylm.shape[0],Ylm.shape[2]))
    for l in range(0,Lmax+1):
        for m in range(-l,l+1):
            Slm[l] += flm[l][m+l]*Ylm[l][m+l]*dA
        Sl[l] = np.linalg.norm(Slm[l])
    return Sl