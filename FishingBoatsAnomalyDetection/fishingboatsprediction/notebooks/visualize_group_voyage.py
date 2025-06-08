import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from db_config import db_config
import random
import sys

# Visualization: plot random sample of voyages with full track and highlighted suspicious segments
# Usage: python visualize_voyage.py [<sample_size>]

# Build DB engine
DATABASE_URL = (
    f"postgresql://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)
engine = create_engine(DATABASE_URL)


def load_trajectory(mmsi: int, eta_time: pd.Timestamp) -> pd.DataFrame:
    """
    Load AIS track for the given voyage.
    Returns DataFrame with columns: timestamp, lat, lon.
    """
    sql = text("""
        SELECT posutc AS ts_string,
               latitude AS lat,
               longitude AS lon
        FROM public.aisdetails
        WHERE mmsi = :m
          AND eta <> 'Unknown'
          AND eta::timestamp = :e
        ORDER BY ts_string
    """ )
    df = pd.read_sql(sql, engine, params={"m": mmsi, "e": eta_time})
    df['timestamp'] = pd.to_datetime(
        df['ts_string'], format="%Y-%m-%d %H:%M:%S", errors='coerce'
    )
    return df.dropna(subset=['timestamp'])[['timestamp', 'lat', 'lon']]


def load_suspicious_segments(mmsi: int, eta_time: pd.Timestamp) -> pd.DataFrame:
    """
    Load suspicious segment intervals for the voyage.
    Returns DataFrame with columns: segment_start, segment_end.
    """
    sql = text("""
        SELECT segment_start, segment_end
        FROM public.eta_suspicious_seg
        WHERE mmsi = :m
          AND eta_time = :e
        ORDER BY segment_start
    """ )
    segs = pd.read_sql(
        sql, engine,
        params={"m": mmsi, "e": eta_time},
        parse_dates=['segment_start', 'segment_end']
    )
    return segs


def plot_voyage(ax, mmsi: int, eta_time: pd.Timestamp):
    """
    Plot voyage trajectory on ax. Full track in blue, suspicious in red.
    """
    traj = load_trajectory(mmsi, eta_time)
    segs = load_suspicious_segments(mmsi, eta_time)

    # full track
    ax.plot(traj['lon'].values, traj['lat'].values,
            color='blue', linewidth=1, label='Track')

    # suspicious segments
    plotted = False
    for _, row in segs.iterrows():
        mask = (
            (traj['timestamp'] >= row['segment_start']) &
            (traj['timestamp'] <= row['segment_end'])
        )
        seg_traj = traj[mask]
        if not seg_traj.empty:
            ax.plot(seg_traj['lon'].values,
                    seg_traj['lat'].values,
                    color='red', linewidth=2,
                    label='Suspicious' if not plotted else '')
            plotted = True

    ax.set_title(f"MMSI={mmsi}, ETA={eta_time}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper left')
    ax.grid(True)


def plot_random_voyages(sample_size: int = 10):
    """
    Randomly sample voyages with anomalies and plot them in a grid.
    """
    # fetch distinct voyages with suspicious segments
    sql = text("SELECT DISTINCT mmsi, eta_time FROM public.eta_suspicious_seg")
    voyages = pd.read_sql(sql, engine, parse_dates=['eta_time'])
    if voyages.empty:
        print("No suspicious voyages found.")
        return

    sample = voyages.sample(n=min(sample_size, len(voyages)), random_state=42).reset_index(drop=True)

    # determine plot grid
    cols = 5
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
    axes_flat = axes.flatten()

    for idx, row in sample.iterrows():
        plot_voyage(axes_flat[idx], int(row['mmsi']), row['eta_time'])

    # hide unused subplots
    for ax in axes_flat[len(sample):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    n = 10
    if len(sys.argv) >= 2:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    plot_random_voyages(n)
