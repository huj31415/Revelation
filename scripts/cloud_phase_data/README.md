# Cloud phase source data

The OPAC cloud tables in this directory were downloaded from the
official OPAC data endpoint (the files are archived here because the live
endpoint currently closes TLS connections):

- `https://cds-espri.ipsl.upmc.fr/espri/pubipsl/aerosols/optdat/cucc00`
- `https://cds-espri.ipsl.upmc.fr/espri/pubipsl/aerosols/optdat/stco00`
- `https://cds-espri.ipsl.upmc.fr/espri/pubipsl/aerosols/optdat/cir200`

Archive copies used for the checked-in files:

- `https://web.archive.org/web/20250117045525id_/https://cds-espri.ipsl.upmc.fr/espri/pubipsl/aerosols/optdat/cucc00`
- `https://web.archive.org/web/20250114232549id_/https://cds-espri.ipsl.upmc.fr/espri/pubipsl/aerosols/optdat/stco00`
- `https://web.archive.org/web/20250117232419id_/https://cds-espri.ipsl.upmc.fr/espri/pubipsl/aerosols/optdat/cir200`

The data originates from OPAC (Optical Properties of Aerosols and Clouds).
The LUT generator accepts `--opac-dir` to use another copy or compatible data
set without changing the script.

The CIE 1931 2-degree color matching functions and standard illuminant D65
were downloaded directly from the CIE open data repository:

- `https://files.cie.co.at/CIE_xyz_1931_2deg.csv`
- `https://files.cie.co.at/CIE_std_illum_D65.csv`
