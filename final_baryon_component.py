# final_baryon.py
import numpy as np
import pandas as pd
import os

G = 4.30091e-6  # kpc (km/s)^2 / Msun

# ----------------------------
# Nested Hernquist potential
# ----------------------------
def phiH(Mb, a):
    """
    Returns a function Phi(r, th) for Hernquist potential
    """
    def Phi_hernquist(r, th=0):
        r = np.asarray(r)
        return -G * Mb / (r + a)
    return Phi_hernquist

# ----------------------------
# Load CSV fits
# ----------------------------
def load_fits(fit_file="hernquist_fits_component.csv"):
    if not os.path.exists(fit_file):
        raise FileNotFoundError(f"{fit_file} not found.")
    return pd.read_csv(fit_file)

# ----------------------------
# Return Hernquist potentials for a galaxy
# ----------------------------
def hernquist_potentials_from_fit(galaxyID, fit_file="hernquist_fits_component.csv"):
    """
    Returns three nested Hernquist potentials:
        - Phi_diskgas
        - Phi_bulge
        - Phi_total
    """
    df = load_fits(fit_file)
    row = df[df["GalaxyID"] == galaxyID]
    if row.empty:
        raise ValueError(f"{galaxyID} not found in fit file")

    M_diskgas = row["M_diskgas"].values[0]
    a_diskgas = row["a_diskgas"].values[0]

    M_bulge   = row["M_bulge"].values[0]
    a_bulge   = row["a_bulge"].values[0]

    Phi_diskgas = phiH(M_diskgas, a_diskgas)
    Phi_bulge   = phiH(M_bulge, a_bulge)
    
    # Total potential = sum of disk+gas + bulge
    def Phi_total(r, th=0):
        return Phi_diskgas(r, th) + Phi_bulge(r, th)

    return Phi_diskgas, Phi_bulge, Phi_total