import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import readData as rd
import pandas as pd

gamma_df = pd.read_csv('data/fit_data/gamma_fits.csv')

def ML_realdata(galaxyID):
    galaxy_row = gamma_df[gamma_df["GalaxyID"] == galaxyID]

    if galaxy_row.empty:
        raise ValueError(f"{galaxyID} not found in gamma_fits.csv")

    gamma_disk   = galaxy_row["Gamma_disk"].values[0]
    error_disk   = galaxy_row["Error_disk"].values[0]
    gamma_bulge  = galaxy_row["Gamma_bulge"].values[0]
    error_bulge  = galaxy_row["Error_bulge"].values[0]

    return gamma_disk, error_disk, gamma_bulge, error_bulge


G = 4.30091e-6


# Hernquist rotation curve model
def vc_hernquist(r, M_b, a):
    return np.sqrt(G * M_b * r) / (r + a)


# Nested Hernquist potential
def phiH(Mb, a):
    def Phi_hernquist(r, th):
        r = np.asarray(r)
        return -G * Mb / (r + a)
    return Phi_hernquist


def fit_galaxy(galaxyID, save_plots=True):

    print(f"\nProcessing galaxy: {galaxyID}")
    
    # 1. Load data
    df_rc, units_rc, distance = rd.get_rc_data(galaxyID)
    gamma_disk, gamma_disk_err, gamma_bulge, gamma_bulge_err = ML_realdata(galaxyID)

    # 2. Arrays
    r = np.asarray(df_rc["Rad"])
    Vdisk = np.asarray(df_rc["Vdisk"])
    Vbul  = np.asarray(df_rc["Vbul"])
    Vgas  = np.asarray(df_rc["Vgas"])

    mask = (r > 0) & np.isfinite(r)
    r = r[mask]
    Vdisk = Vdisk[mask]
    Vbul = Vbul[mask]
    Vgas = Vgas[mask]

    # 3. Scale components
    Vbul_scaled  = np.sqrt(gamma_bulge) * Vbul
    Vdisk_scaled = np.sqrt(gamma_disk) * Vdisk
    Vgas_scaled  = Vgas

    Vdiskgas = np.sqrt(Vdisk_scaled**2 + Vgas_scaled**2)

    # Detect bulge
    has_bulge = not np.allclose(Vbul_scaled, 0)

    # 4. Fit Hernquist to Disk+Gas
    popt_dg, _ = curve_fit(
        vc_hernquist,
        r,
        Vdiskgas,
        p0=[6e10, 5],
        bounds=([0, 0], [np.inf, np.inf])
    )

    M_dg_fit, a_dg_fit = popt_dg
    Phi_dg_func = phiH(M_dg_fit, a_dg_fit)

    # 5. Fit Hernquist to Bulge (ONLY if bulge exists)
    if has_bulge:
        popt_b, _ = curve_fit(
            vc_hernquist,
            r,
            Vbul_scaled,
            p0=[1e10, 1],
            bounds=([0, 0], [np.inf, np.inf])
        )
        M_b_fit, a_b_fit = popt_b
        Phi_b_func = phiH(M_b_fit, a_b_fit)
    else:
        M_b_fit, a_b_fit = 0, 1
        Phi_b_func = lambda r, th: np.zeros_like(r)

    # 6. Model grid
    r_model = np.linspace(r.min(), r.max(), 500)

    V_dg_model = vc_hernquist(r_model, M_dg_fit, a_dg_fit)
    V_b_model  = vc_hernquist(r_model, M_b_fit, a_b_fit) if has_bulge else np.zeros_like(r_model)

    V_bul_i = np.interp(r_model, r, Vbul_scaled)

    # rot curve

    fig1 = plt.figure(figsize=(8,5))

    plt.plot(r, Vdiskgas, 'o', label="Disk+Gas Data")
    plt.plot(r, Vbul_scaled, 'o', label="Bulge Data")

    plt.plot(r_model, V_dg_model, '--', label="Hernquist Disk+Gas", linewidth=2)

    if has_bulge:
        plt.plot(r_model, V_b_model, '--', label="Hernquist Bulge", linewidth=2)

    plt.xlabel(f"Radius ({units_rc.at[0,'Rad']})")
    plt.ylabel("Velocity (km/s)")
    plt.title(f"{galaxyID} Rotation Curve Components")
    plt.legend()
    plt.grid()

    # potential

    Phi_dg_model = Phi_dg_func(r_model, 0)
    Phi_b_model  = Phi_b_func(r_model, 0)

    Phi_total = Phi_dg_model + Phi_b_model

    fig2 = plt.figure(figsize=(8,5))

    plt.plot(r_model, Phi_dg_model, label="Hernquist Disk+Gas")
    if has_bulge:
        plt.plot(r_model, Phi_b_model, label="Hernquist Bulge")

    plt.plot(r_model, Phi_total, linewidth=3, label="Total Potential",color='black',ls='--')

    plt.xlabel("Radius (kpc)")
    plt.ylabel(r"Potential ($km^2 s^{-2}$)")
    plt.title(f"{galaxyID} Potential Components")
    plt.legend()
    plt.grid()

    # 2 folders

    if save_plots:

        base_path = os.getcwd()
        main_folder = os.path.join(base_path, "baryon_fit_components")
        subfolder = "with_bulge" if has_bulge else "no_bulge"

        save_folder = os.path.join(main_folder, subfolder)
        os.makedirs(save_folder, exist_ok=True)

        rc_filename  = os.path.join(save_folder, f"{galaxyID}_RC.png")
        pot_filename = os.path.join(save_folder, f"{galaxyID}_potential.png")

        fig1.savefig(rc_filename)
        fig2.savefig(pot_filename)

        print(f"Saved plots for {galaxyID} in {subfolder}/")

    plt.close(fig1)
    plt.close(fig2)

    return {
        "galaxyID": galaxyID,
        "M_dg": M_dg_fit,
        "a_dg": a_dg_fit,
        "M_b": M_b_fit,
        "a_b": a_b_fit
    }


def run_all_galaxies():

    galaxy_list = rd.get_galaxy_ids()
    results = []

    for gal in galaxy_list:
        try:
            result = fit_galaxy(gal)
            results.append(result)
        except Exception as e:
            print(f"{gal} failed: {e}")

    return results


if __name__ == "__main__":
    run_all_galaxies()