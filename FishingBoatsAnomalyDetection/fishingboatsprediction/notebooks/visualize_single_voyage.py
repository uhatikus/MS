import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from db_config import db_config

# Visualization: plot full voyage with suspicious segments highlighted
# Usage: python visualize_voyage.py <mmsi> '<eta_time>'

# Build DB engine
DATABASE_URL = (
    f"postgresql://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)
engine = create_engine(DATABASE_URL)


def load_trajectory(mmsi: int, eta_time: pd.Timestamp) -> pd.DataFrame:
    """
    Load AIS track for the given voyage from aisdetails.
    Returns DataFrame with columns: timestamp, lat, lon.
    """
    sql = text("""
        SELECT 
            posutc       AS ts_string,
            latitude     AS lat,
            longitude    AS lon
        FROM public.aisdetails
        WHERE mmsi = :m
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
    return pd.read_sql(
        sql, engine,
        params={"m": mmsi, "e": eta_time},
        parse_dates=['segment_start', 'segment_end']
    )


def plot_voyage(mmsi: int, eta_time: str):
    """
    Plot voyage trajectory. Normal track in blue, suspicious segments in red.
    """
    eta_ts = pd.to_datetime(eta_time)
    traj = load_trajectory(mmsi, eta_ts)
    segs = load_suspicious_segments(mmsi, eta_ts)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot normal track first
    ax.plot(traj['lon'].values, traj['lat'].values,
            color='blue', linewidth=1, label='Normal')

    # Overlay suspicious segments
    for _, row in segs.iterrows():
        mask = (
            (traj['timestamp'] >= row['segment_start']) &
            (traj['timestamp'] <= row['segment_end'])
        )
        seg_traj = traj[mask]
        if not seg_traj.empty:
            ax.plot(
                seg_traj['lon'].values,
                seg_traj['lat'].values,
                color='red', linewidth=2,
                label='Suspicious' if 'Suspicious' not in ax.get_legend_handles_labels()[1] else ''
            )

    ax.set_title(f"Voyage MMSI={mmsi}, ETA={eta_time}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python visualize_voyage.py <mmsi> '<eta_time>'")
        sys.exit(1)
    mmsi_arg = int(sys.argv[1])
    eta_arg = sys.argv[2]
    plot_voyage(mmsi_arg, eta_arg)
