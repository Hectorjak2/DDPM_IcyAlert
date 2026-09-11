"""To use the api one has to
1. Have an account and log in

2. Go to the data page and accept Terms of Use: https://cds.climate.copernicus.eu/datasets/reanalysis-pan-carra-means?tab=overview

3. Copy the api url and key from this page: https://cds.climate.copernicus.eu/how-to-api

4. Create a file called ".cdsapirc" containing the url and key in ones home/user folder. The file should have two lines.
For linux and macos: 
cat > $HOME/.cdsapirc << 'EOF' url: 'YOURURL' key: 'YOURKEY' EOF
    
5. You may be missing eccodes and cfgrib, which are needed to read the grib files. You can install them using brew install eccodes and pip install cfgrib. 
"""

import cdsapi
import os
import xarray as xr

def download_carra2_monthly_data(save_name: str, years: list[str]):
    ds_name = "CARRA2_DAILY"
    CARRA_PAN_AREA = [90, -180, 40, 180]   # full pan-CARRA domain
    dataset = "reanalysis-pan-carra-means"
    request = {
        "time_aggregation": "daily",
        "level_type": "single_levels",
        "product_type": "analysis_based",
        "year": [
            "2000", "2001", "2002",
            "2003", "2004", "2005",
            "2006", "2007", "2008",
            "2009", "2010", "2011",
            "2012", "2013", "2014",
            "2015", "2016", "2017",
            "2018", "2019", "2020",
            "2021", "2022", "2023",
            "2024", "2025"
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "data_format": "grib",
        "variable": ["sea_ice_area_fraction"],
        "area": [88, -90, 56, 30]
    }

    client = cdsapi.Client()

    grib_file = f"./data/{ds_name}/all.grib"
    os.makedirs(os.path.dirname(grib_file), exist_ok=True)

    client.retrieve(dataset, request).download(grib_file)
    print(f"Downloaded CARRA2 data to {grib_file}")

    ds = xr.open_dataset(grib_file, engine='cfgrib')
    save_path = f'./data/{ds_name}/{save_name}.zarr'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ds.to_zarr(save_path, mode="w")
    print(f"Converted GRIB file to Zarr format at {save_path}")

    os.remove(grib_file)
    print(f"Deleted temporary GRIB file {grib_file}")

if __name__ == "__main__":
    download_carra2_monthly_data("dataset", [f"20{i:02d}" for i in range(24)])
