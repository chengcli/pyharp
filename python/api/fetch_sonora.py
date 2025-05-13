#! /usr/bin/env python3
import requests
import argparse
from tqdm import tqdm

"""
DATA URL: https://zenodo.org/records/5590989#.Yzy1qezMJb9

There are 108 correlated k-coefficients datasets, using the naming convention sonora_2020_fehxxxx_co_yyy.data.196.tar.gz (54 models) or sonora_2020_fehxxxx_co_yyy_noTiOVO.data.196.tar.gz (54 models), where xxxx is the metallicity in 10x dex relative to solar, and yyy is the 100x C/O ratio relative to solar, as a multiplication factor. For example a metallicity of +000 and a C/O ratio of 100 indicates solar abundances, feh+070 should be read as a metallicity of +0.7 dex, feh-100 as -1.0 dex, co_025 should be read as 0.25x C/O relative to solar, and co_200 as 2x C/O relative to solar. We use the Lodders et al. 2010 value for the solar C/O=0.458. The files with “noTiOVO” in the filename contain the correlated k-coefficients calculated without the opacity of TiO and VO. The rest of the molecular abundances and opacities are the same as the equilibrium chemistry values in the regular files. These files are useful for calculating models without any TiO- and VO-induced temperature inversion in the atmosphere.

The correlated-k coefficients are calculated using pre-mixed opacities, with abundances given by equilibrium chemistry for each metallicity-C/O combination, as described in Marley et al. 2021. There are 9 Fe/H values: 0.0, 0.3, 0.5, 0.7, 1.0, -0.3, -0.5, -0.7, and -1.0; and 6 C/O values: 0.25, 0.5, 1.0, 1.5, 2.0 and 2.5. The k-coefficients are calculated for a grid of 1460 pressure-temperature points, from10^−6 to 3000 bar and from 75 to 4000 K, listed in the file 1460_layer_list, and can be read in using the IDL script read_k_coefficients_1460.pro. The spectral windows are listed in the file 196_windows.txt (intervals defined as starting at lambda1 and ending at lambda2).

The opacity sources included in the calculations are: C2H2, C2H4, C2H6, CH4, CO, CO2, CrH, Fe, FeH, H2, H3+, H2O, H2S, HCN, LiCl, LiF, LiH, MgH, N2, NH3, OCS, PH3, SiO, TiO, and VO, in addition to alkali metals (Li, Na, K, Rb, Cs). The corresponding high resolution opacities for these atoms and molecules can be found in the Zenodo repository 10.5281/zenodo.6600976. The references for the line lists used in these opacity calculations are listed in the file Opacity_references_2021.pdf. Please include these references, as well as the reference to this Zenodo repository when publishing your paper.
"""

# version3
sonora2020_url_tmplate = "https://zenodo.org/records/7542068/files/sonora_2020_feh{feh}_co_{co}.data.196.tar.gz?download=1"

valid_feh = [
    "+000", "+030", "+050", "+070", "+100", "-030", "-050", "-070", "-100"
]

valid_co = [
    "025", "050", "100", "150", "200", "250"
]

def download_file(url, filename=None):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error for bad status

    total_size = int(response.headers.get("Content-Length", 0))
    if filename is None:
        filename = url.split("/")[-1]  # Use last part of URL as default filename

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Downloaded to {filename}")

def get_sonora_data(feh: str, co: str):
    if feh not in valid_feh:
        raise ValueError(f"Invalid feh value: {feh}. Valid values are: {valid_feh}")
    if co not in valid_co:
        raise ValueError(f"Invalid co value: {co}. Valid values are: {valid_co}")

    return sonora2020_url_tmplate.format(feh=feh, co=co)

def main():
    parser = argparse.ArgumentParser(description="Download Sonora 2020 data files.")
    parser.add_argument(
        "--feh",
        default="+000",
        choices=valid_feh,
        type=str,
        help="Metallicity in 10x dex relative to solar (e.g., +000, -100), default +000",
    )
    parser.add_argument(
        "--co",
        default="100",
        choices=valid_co,
        type=str,
        help="C/O ratio relative to solar (e.g., 100, 200), default 100",
    )

    args = parser.parse_args()
    print(f"Download Sonora 2020 data file: feh = {args.feh}, co = {args.co}")

    url = get_sonora_data(args.feh, args.co)
    download_file(url, filename=f"sonora_2020_feh{args.feh}_co_{args.co}.data.196.tar.gz")

if __name__ == "__main__":
    main()
