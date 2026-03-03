import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import readData as rd

G = 4.30091e-6  # kpc (km/s)^2 / Msun


#1 Gamma data

gamma_df = pd.read_csv('data/fit_data/gamma_fits.csv')

def get_gamma(galaxyID):
    row = gamma_df[gamma_df["GalaxyID"] == galaxyID]
    if row.empty:
        raise ValueError(f"{galaxyID} not found in gamma_fits.csv")

    return (
        row["Gamma_disk"].values[0],
        row["Gamma_bulge"].values[0])


# 2 Baryon velocity

def compute_baryon_velocity(r, Vdisk, Vbul, Vgas,
                            gamma_disk,
                            gamma_bulge,
                            include_bulge=True):

    if not include_bulge:
        gamma_bulge = 0.0

    Vb = np.sqrt(
        gamma_disk * Vdisk**2 +
        gamma_bulge * Vbul**2 +
        Vgas**2)

    return Vb


# 3 Hernquist

def vc_hernquist(r, M_b, a):
    return np.sqrt(G * M_b * r) / (r + a)


# 4 Galaxy fitting

def fit_single_galaxy(galaxyID):

    df_rc, units_rc, distance = rd.get_rc_data(galaxyID)
    gamma_disk, gamma_bulge = get_gamma(galaxyID)

    r = np.asarray(df_rc["Rad"])
    Vdisk = np.asarray(df_rc["Vdisk"])
    Vbul  = np.asarray(df_rc["Vbul"])
    Vgas  = np.asarray(df_rc["Vgas"])

    mask = (r > 0) & np.isfinite(r)
    r, Vdisk, Vbul, Vgas = r[mask], Vdisk[mask], Vbul[mask], Vgas[mask]

    # ----------------------------
    # Disk + Gas component
    # ----------------------------
    V_diskgas = np.sqrt(
        gamma_disk * Vdisk**2 +
        Vgas**2
    )

    # ----------------------------
    # Bulge component
    # ----------------------------
    V_bulge = np.sqrt(
        gamma_bulge * Vbul**2
    )

    p0 = [6e10, 5]
    bounds = ([0, 0], [np.inf, np.inf])

    # Fit disk+gas
    popt_diskgas, _ = curve_fit(
        vc_hernquist, r, V_diskgas,
        p0=p0, bounds=bounds
    )

    M_diskgas, a_diskgas = popt_diskgas

    # Fit bulge (only if non-zero)
    if np.any(V_bulge > 0):

        popt_bulge, _ = curve_fit(
            vc_hernquist, r, V_bulge,
            p0=p0, bounds=bounds
        )

        M_bulge, a_bulge = popt_bulge

    else:
        M_bulge, a_bulge = 0.0, 0.0

    return {
        "GalaxyID": galaxyID,
        "M_diskgas": M_diskgas,
        "a_diskgas": a_diskgas,
        "M_bulge": M_bulge,
        "a_bulge": a_bulge
    }


# 5 Repeat for all galaxies

def run_fits_and_save(output_file="hernquist_fits_component.csv",
                      include_bulge=True,
                      only_no_bulge=False):

    galaxy_list = rd.get_galaxy_ids()
    results = []

    for gal in galaxy_list:
        try:

            # Optional: filter galaxies with no bulge
            if only_no_bulge:
                df_rc, _, _ = rd.get_rc_data(gal)
                if np.all(np.asarray(df_rc["Vbul"]) == 0):
                    pass
                else:
                    continue

            result = fit_single_galaxy(
                gal)

            results.append(result)

        except Exception as e:
            print(f"{gal} failed: {e}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

    print(f"Saved fitted parameters to {output_file}")
    return df_results


# 7 Main function

if __name__ == "__main__":
    run_fits_and_save()
