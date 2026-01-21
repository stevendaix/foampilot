import pandas as pd
import csv
import plotly.express as px


class WeatherFileEPW:
    """
    A class to manage EnergyPlus Weather (EPW) files.

    This class provides functionalities for reading, writing, analyzing, and visualizing climate data.

    Attributes
    ----------
    headers : dict
        Dictionary storing the EPW file headers.
    dataframe : pd.DataFrame
        DataFrame containing the climate data.
    """

    def __init__(self):
        """Initialize the WeatherFileEPW instance with empty headers and DataFrame."""
        self.headers = {}
        self.dataframe = pd.DataFrame()

    def read(self, fp: str):
        """
        Read headers and climate data from an EPW file.

        Parameters
        ----------
        fp : str
            File path of the EPW file to read.
        """
        self.headers = self._read_headers(fp)
        self.dataframe = self._read_data(fp)

    def _read_headers(self, fp: str) -> dict:
        """
        Read headers from an EPW file.

        Parameters
        ----------
        fp : str
            File path of the EPW file.

        Returns
        -------
        dict
            Dictionary containing header information. Keys are header identifiers, values are lists of metadata.
        """
        headers = {}
        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if row[0].isdigit():
                    break
                headers[row[0]] = row[1:]
        return headers

    def _read_data(self, fp: str) -> pd.DataFrame:
        """
        Read climate data from an EPW file into a DataFrame.

        Parameters
        ----------
        fp : str
            File path of the EPW file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing climate data with columns corresponding to EPW fields.
        """
        column_names = [
            'Year', 'Month', 'Day', 'Hour', 'Minute',
            'Data Source and Uncertainty Flags', 'Dry Bulb Temperature',
            'Dew Point Temperature', 'Relative Humidity',
            'Atmospheric Station Pressure', 'Extraterrestrial Horizontal Radiation',
            'Extraterrestrial Direct Normal Radiation', 'Horizontal Infrared Radiation Intensity',
            'Global Horizontal Radiation', 'Direct Normal Radiation',
            'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance',
            'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance',
            'Zenith Luminance', 'Wind Direction', 'Wind Speed',
            'Total Sky Cover', 'Opaque Sky Cover',
            'Visibility', 'Ceiling Height', 'Present Weather Observation',
            'Present Weather Codes', 'Precipitable Water',
            'Aerosol Optical Depth', 'Snow Depth', 'Days Since Last Snowfall',
            'Albedo', 'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'
        ]
        first_row = self._first_row_with_climate_data(fp)
        return pd.read_csv(fp, skiprows=first_row, header=None, names=column_names)

    def _first_row_with_climate_data(self, fp: str) -> int:
        """
        Find the first row in the EPW file that contains climate data.

        Parameters
        ----------
        fp : str
            File path of the EPW file.

        Returns
        -------
        int
            Row index where climate data starts.
        """
        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, row in enumerate(csvreader):
                if row[0].isdigit():
                    return i
        return -1

    def write(self, fp: str):
        """
        Write the EPW file headers and climate data to a new file.

        Parameters
        ----------
        fp : str
            File path to save the EPW file.
        """
        with open(fp, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for k, v in self.headers.items():
                csvwriter.writerow([k] + v)
            for row in self.dataframe.itertuples(index=False):
                csvwriter.writerow(row)

    def analyze_temperature(self) -> pd.Series:
        """
        Calculate statistics for dry bulb temperature.

        Returns
        -------
        pd.Series
            Statistical summary (mean, std, min, max, etc.) of dry bulb temperature.

        Raises
        ------
        ValueError
            If no climate data has been loaded.
        """
        if not self.dataframe.empty:
            return self.dataframe['Dry Bulb Temperature'].describe()
        else:
            raise ValueError("No data loaded. Please read an EPW file first.")

    def analyze_humidity(self) -> pd.Series:
        """
        Calculate statistics for relative humidity.

        Returns
        -------
        pd.Series
            Statistical summary (mean, std, min, max, etc.) of relative humidity.

        Raises
        ------
        ValueError
            If no climate data has been loaded.
        """
        if not self.dataframe.empty:
            return self.dataframe['Relative Humidity'].describe()
        else:
            raise ValueError("No data loaded. Please read an EPW file first.")

    def plot_wind_rose(self):
        """
        Generate a wind rose plot using wind speed and direction.

        Raises
        ------
        ValueError
            If no climate data has been loaded.
        """
        if not self.dataframe.empty:
            fig = px.scatter_polar(
                self.dataframe,
                r="Wind Speed",
                theta="Wind Direction",
                color="Wind Speed",
                title="Wind Rose",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig.show()
        else:
            raise ValueError("No data loaded. Please read an EPW file first.")

    def get_header(self) -> dict:
        """
        Return the EPW file headers.

        Returns
        -------
        dict
            Headers dictionary.

        Raises
        ------
        ValueError
            If headers have not been loaded.
        """
        if self.headers:
            return self.headers
        else:
            raise ValueError("Headers not loaded. Please read an EPW file first.")
    def compute_wind_frequencies(
    self,
    direction_bin: float = 22.5,
    speed_bins: list = None,
    min_speed: float = 0.0,
):
    """
    Compute wind frequency table from EPW data.

    Parameters
    ----------
    direction_bin : float
        Angular resolution for wind direction bins (deg).
        Typical values: 30°, 22.5°, 15°.
    speed_bins : list
        Wind speed bins (m/s), e.g. [0, 2, 4, 6, 8, 10, 15].
        If None, Lawson-relevant bins are used.
    min_speed : float
        Minimum wind speed to consider (filters calm winds).

    Returns
    -------
    dict
        Wind rose frequencies:
        {
            direction_deg: [
                {"speed": speed_bin_center, "frequency": freq},
                ...
            ]
        }
    """
    if self.dataframe.empty:
        raise ValueError("No climate data loaded. Please read an EPW file first.")

    df = self.dataframe.copy()

    if speed_bins is None:
        speed_bins = [0, 2, 4, 6, 8, 10, 15, 25]

    # Filter calm winds
    df = df[df["Wind Speed"] >= min_speed]

    # Direction binning
    df["Dir_bin"] = (
        (df["Wind Direction"] / direction_bin).round() * direction_bin
    ) % 360

    # Speed binning
    df["Speed_bin"] = pd.cut(
        df["Wind Speed"],
        bins=speed_bins,
        right=False,
        include_lowest=True,
    )

    total_count = len(df)
    if total_count == 0:
        raise ValueError("No wind data after filtering.")

    freq_table = {}

    grouped = df.groupby(["Dir_bin", "Speed_bin"]).size().reset_index(name="count")

    for _, row in grouped.iterrows():
        direction = float(row["Dir_bin"])
        speed_interval = row["Speed_bin"]
        count = row["count"]

        speed_center = 0.5 * (speed_interval.left + speed_interval.right)
        frequency = count / total_count

        if direction not in freq_table:
            freq_table[direction] = []

        freq_table[direction].append(
            {
                "speed": float(speed_center),
                "frequency": float(frequency),
            }
        )

    return freq_table

    def export_wind_frequencies(self, output_file: str, **kwargs):
    freq = self.compute_wind_frequencies(**kwargs)
    with open(output_file, "w") as f:
        json.dump(freq, f, indent=4)


    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the DataFrame containing climate data.

        Returns
        -------
        pd.DataFrame
            Climate data DataFrame.

        Raises
        ------
        ValueError
            If climate data has not been loaded.
        """
        if not self.dataframe.empty:
            return self.dataframe
        else:
            raise ValueError("Data not loaded. Please read an EPW file first.")