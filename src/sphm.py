import numpy as np
from scipy.special import sph_harm

# Compute all spherial harmonics Ylm for a given Lmax
def compute_all_harmonics(theta,phi,Lmax):
    
    Ylm_r = np.zeros((Lmax+1, 2*Lmax+1, len(theta)))
    Ylm_i = np.zeros((Lmax+1, 2*Lmax+1, len(theta)))
    Ylm = np.zeros((Lmax+1, 2*Lmax+1, len(theta)))

    for l in range(0,Lmax+1):
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

# Spherical harmonics decomposition of a complex function with real part f and imaginary part fi (fi=0 if the function is real)
def compute_decomposition(f,fi,Ylm_r,Ylm_i,dA,Lmax):
    flm_r = []
    flm_i = []
    flm = []
    surface_area = np.sum(dA)
    for l in range(0,Lmax+1):
      flm_r += [[]]
      flm_i += [[]]
      flm += [[]]

      for m in range(-l, l+1):
        flm_r[l] += [4*np.pi * np.sum(dA[:,] * (f[:,] * Ylm_r[l,m+l,:] + fi[:,] * Ylm_i[l,m+l,:]))/(surface_area)]
        flm_i[l] += [4*np.pi * np.sum(dA[:,] * (fi[:,] * Ylm_r[l,m+l,:] - f[:,] * Ylm_i[l,m+l,:]))/(surface_area)]

      for m in range(-l, l+1):
        flm[l] += [flm_r[l][l+m]]
        if(m>0):
          flm[l][m+l] = flm_r[l][m+l]*np.sqrt(2)
        elif(m<0):
          flm[l][m+l] = flm_i[l][-m+l]*np.sqrt(2)

    return flm_r, flm_i, flm

# Rotation-invariant representation of harmonics
def compute_energy_representation(flm, Ylm, Lmax, dA):
    Sl = np.zeros((Lmax+1))
    Slm = np.zeros((Ylm.shape[0],Ylm.shape[2]))
    for l in range(0,Lmax+1):
        for m in range(-l,l+1):
            Slm[l] += flm[l][m+l]*Ylm[l][m+l]*dA
        Sl[l] = np.round(np.linalg.norm(Slm[l]),2)
    return Sl

# Compute Spherical harmonics of a field on a mesh
def compute_sphm_inv_rep(f, fi, theta, phi, dA, Lmax):
   #_, theta, phi, dA = compute_vertex_properties(mesh)
   Ylm_r,Ylm_i, Ylm = compute_all_harmonics(theta,phi,Lmax)
   _, _, flm = compute_decomposition(f,fi,Ylm_r,Ylm_i,dA,Lmax)
   Sl = compute_energy_representation(flm, Ylm, Lmax, dA)
   return Sl