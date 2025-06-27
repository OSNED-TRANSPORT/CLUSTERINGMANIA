import os
import logging
import pandas as pd
import requests
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
import streamlit as st
from streamlit_folium import st_folium

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

###############################################################################
# SETTINGS AND FILE PATHS
###############################################################################
LOCATIONS_FILE = r"C:\Users\v.kok\Desktop\Denso\Simulation\MASTER\DISTANCE_MATRIX\locationsanalysis.xlsx"
OUTPUT_FILE = r"C:\Users\v.kok\Desktop\Denso\Simulation\MASTER\CLUSTER_ROUTE\cluster_assignmentsKMEANS.xlsx"
# New clusters file path at the same location as LOCATIONS_FILE
CLUSTERS_FILE = os.path.join(os.path.dirname(LOCATIONS_FILE), "CLUSTERS.xlsx")

# Use direct demand weighting instead of inverse demand.
USE_DEMAND_WEIGHTING = True
DEMAND_WEIGHT_FACTOR = 100.0

# Dummy variable to control number of clusters (cross dock locations)
N_CLUSTERS = 1
THRESHOLD = 35  # Not used in KMeans, but preserved for consistency of settings
MAX_ASSIGNMENT_PER_CLUSTER = 68  # Not used in KMeans since clusters are fixed
OVERPASS_RADIUS = 50000

# New settings for snapping to highway
REQUIRE_HIGHWAY = True
HIGHWAY_SEARCH_RADIUS = 100000  # in meters

# NEW: Dummy variable to scale the normalized demand feature.
# Changing this value should affect clustering if it is set up correctly.
DUMMY_DEMAND_WEIGHT = 100.0

# NEW: Dummy variable to adjust the weight for the geo locations.
# Multiplying the normalized longitude and latitude dimensions.
DUMMY_GEO_WEIGHT = 1


###############################################################################
# HELPER FUNCTIONS
###############################################################################
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))


def load_locations_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    for col in ["ALL_LOCATION_GID", "Lat", "Long"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    if USE_DEMAND_WEIGHTING:
        if "demand" not in df.columns:
            raise ValueError("Demand weighting enabled, but 'demand' column is missing.")
        # Read direct demand; you can later experiment with transformations (e.g., logarithmic)
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0)
    return df


def compute_centroid(df):
    return df["Lat"].mean(), df["Long"].mean()


def calc_display_label(label):
    return label + 1 if label != -1 else "Noise"


# Function to compute weighted centroid using direct demand as the weight.
# This calculates the centroid using a weighted average of the locations.
def compute_weighted_centroid(df_cluster):
    weights = df_cluster["demand"]
    if weights.sum() == 0:
        return df_cluster["Lat"].mean(), df_cluster["Long"].mean()
    weighted_lat = np.average(df_cluster["Lat"], weights=weights)
    weighted_long = np.average(df_cluster["Long"], weights=weights)
    return weighted_lat, weighted_long


# Function to fetch logistical hubs using Overpass API.
def fetch_logistical_hubs(lat, lon, radius=OVERPASS_RADIUS):
    query = f"""
    [out:json];
    (
      node["building"="warehouse"](around:{radius},{lat},{lon});
      way["building"="warehouse"](around:{radius},{lat},{lon});
      relation["building"="warehouse"](around:{radius},{lat},{lon});
    );
    out center;
    """
    url = "http://overpass-api.de/api/interpreter"
    try:
        response = requests.get(url, params={'data': query}, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Overpass API error (hubs): {e}")
        return []
    elements = response.json().get("elements", [])
    hubs = []
    for el in elements:
        if el["type"] == "node":
            hubs.append({"id": el["id"], "lat": el["lat"], "lon": el["lon"], "tags": el.get("tags", {})})
        elif "center" in el:
            hubs.append({"id": el["id"], "lat": el["center"]["lat"], "lon": el["center"]["lon"],
                         "tags": el.get("tags", {})})
    return hubs


# Function to fetch highways using Overpass API.
def fetch_highways(lat, lon, radius=HIGHWAY_SEARCH_RADIUS):
    query = f"""
    [out:json];
    (
      node["highway"~"motorway|trunk|primary"](around:{radius},{lat},{lon});
    );
    out center;
    """
    url = "http://overpass-api.de/api/interpreter"
    try:
        response = requests.get(url, params={'data': query}, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Overpass API error (highways): {e}")
        return []
    elements = response.json().get("elements", [])
    highways = []
    for el in elements:
        if el["type"] == "node":
            highways.append({"id": el["id"], "lat": el["lat"], "lon": el["lon"], "tags": el.get("tags", {})})
    return highways


# Function to find the nearest feature (logistical hub or highway) to a given centroid.
def find_nearest_feature(centroid, features):
    best_feature, best_distance = None, float("inf")
    for feature in features:
        d = haversine_distance(centroid[0], centroid[1], feature["lat"], feature["lon"])
        if d < best_distance:
            best_distance = d
            best_feature = feature
    return best_feature, best_distance


###############################################################################
# KMEANS CLUSTERING FOR SUPPLIERS WITH NORMALIZATION & WEIGHTED CENTROID RETUNING
# USING DIRECT DEMAND FOR WEIGHTING
###############################################################################
def get_cluster_data():
    # If an output file exists, load cluster assignments.
    if os.path.exists(OUTPUT_FILE):
        logging.info(f"Loading cluster assignments from {OUTPUT_FILE}")
        return pd.read_excel(OUTPUT_FILE)

    logging.info("Computing cluster assignments using KMeans...")
    df = load_locations_data(LOCATIONS_FILE)
    df = df.rename(columns={"ALL_LOCATION_GID": "location_id"}).dropna(subset=["Lat", "Long"])

    # Prepare features for KMeans. Now use direct demand (rather than inverse) for weighting.
    if USE_DEMAND_WEIGHTING:
        # Scale the direct demand feature by a factor.
        df["weighted_demand"] = df["demand"] * DEMAND_WEIGHT_FACTOR
    else:
        df["weighted_demand"] = 0

    # Create a feature set: [Long, Lat, weighted_demand]
    features = df[["Long", "Lat", "weighted_demand"]].values

    # Normalize the features so they all have similar ranges.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Multiply the third column (demand) by DUMMY_DEMAND_WEIGHT to control its contribution.
    features_scaled[:, 2] *= DUMMY_DEMAND_WEIGHT
    # Multiply the first two columns (geographic coordinates) by DUMMY_GEO_WEIGHT to control their weight.
    features_scaled[:, :2] *= DUMMY_GEO_WEIGHT

    # Run KMeans clustering with normalized features.
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    df["final_cluster"] = kmeans.fit_predict(features_scaled)
    df["display_cluster"] = df["final_cluster"].apply(calc_display_label)

    # Compute cluster centers for the first two dimensions (Long and Lat) from the scaled space.
    centers_scaled = kmeans.cluster_centers_[:, :2]
    # Inverse transform: first set the 3rd dimension to zero then inverse transform and select only Long and Lat.
    centers_unscaled = scaler.inverse_transform(
        np.column_stack((centers_scaled, np.zeros(centers_scaled.shape[0])))
    )[:, :2]
    init_centroid_dict = {clu: (centers_unscaled[clu][1], centers_unscaled[clu][0]) for clu in range(N_CLUSTERS)}

    # Recalculate centroids for each cluster using a weighted average with direct demand as weight.
    weighted_centroid_dict = {}
    for clu in range(N_CLUSTERS):
        df_cluster = df[df["final_cluster"] == clu]
        weighted_centroid_dict[clu] = compute_weighted_centroid(df_cluster)

    # Use the weighted centroids as final centroids.
    df["centroid_lat"] = df["final_cluster"].map(lambda x: weighted_centroid_dict[x][0])
    df["centroid_long"] = df["final_cluster"].map(lambda x: weighted_centroid_dict[x][1])

    # Snap centers to nearest logistical hub using weighted centroids.
    snap_hub_dict = {}
    for clu, centroid in weighted_centroid_dict.items():
        hubs = fetch_logistical_hubs(centroid[0], centroid[1])
        nearest_hub, _ = find_nearest_feature(centroid, hubs)
        snap_hub_dict[clu] = (nearest_hub["lat"], nearest_hub["lon"]) if nearest_hub else centroid

    df["snapped_lat_hub"] = df["final_cluster"].apply(lambda clu: snap_hub_dict.get(clu, (None, None))[0])
    df["snapped_lon_hub"] = df["final_cluster"].apply(lambda clu: snap_hub_dict.get(clu, (None, None))[1])

    # If highway snapping is required, compute snapped highway positions using weighted centroids.
    if REQUIRE_HIGHWAY:
        snap_highway_dict = {}
        for clu, centroid in weighted_centroid_dict.items():
            highways = fetch_highways(centroid[0], centroid[1])
            nearest_highway, _ = find_nearest_feature(centroid, highways)
            snap_highway_dict[clu] = (nearest_highway["lat"], nearest_highway["lon"]) if nearest_highway else centroid
        df["snapped_lat_highway"] = df["final_cluster"].apply(lambda clu: snap_highway_dict.get(clu, (None, None))[0])
        df["snapped_lon_highway"] = df["final_cluster"].apply(lambda clu: snap_highway_dict.get(clu, (None, None))[1])
    else:
        df["snapped_lat_highway"] = df["centroid_lat"]
        df["snapped_lon_highway"] = df["centroid_long"]

    # Save the final cluster assignments with weighted centroid and snapping info.
    df.to_excel(OUTPUT_FILE, index=False)
    logging.info(f"Cluster assignments saved to {OUTPUT_FILE}")
    return df


df_locations = get_cluster_data()

# Create a separate Excel file to save snapped geographical cluster locations.
clusters_df = df_locations.groupby("final_cluster", as_index=False).agg({
    "snapped_lat_hub": "first",
    "snapped_lon_hub": "first",
    "snapped_lat_highway": "first",
    "snapped_lon_highway": "first",
    "location_id": "count"
}).rename(columns={
    "snapped_lat_hub": "snapped_lat_hub",
    "snapped_lon_hub": "snapped_lon_hub",
    "snapped_lat_highway": "snapped_lat_highway",
    "snapped_lon_highway": "snapped_lon_highway",
    "location_id": "supplier_count"
})
clusters_df.to_excel(CLUSTERS_FILE, index=False)
logging.info(f"Clusters geographical locations saved to {CLUSTERS_FILE}")

###############################################################################
# VISUALIZATION WITH FOLIUM & STREAMLIT
###############################################################################
st.set_page_config(layout="wide", page_title="Supplier Clusters with KMeans")
st.title("Supplier Clusters Interactive Map (KMeans)")

cluster_options = ["All Clusters", "Noise"] + sorted(
    [str(opt) for opt in set(df_locations[df_locations["display_cluster"] != "Noise"]["display_cluster"])],
    key=lambda x: int(x))
selected_cluster = st.sidebar.selectbox("Select Cluster to Display", cluster_options)

if selected_cluster == "All Clusters":
    df_filtered = df_locations.copy()
elif selected_cluster == "Noise":
    df_filtered = df_locations[df_locations["final_cluster"] == -1]
else:
    cluster_num = int(selected_cluster) - 1
    df_filtered = df_locations[df_locations["final_cluster"] == cluster_num]

cluster_colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "darkblue",
                  "darkgreen", "cadetblue", "darkpurple", "pink", "lightblue", "lightgreen", "gray",
                  "black", "lightgray"]

# Build map based on filtered clusters (all clusters view)
if selected_cluster == "All Clusters":
    center_lat = df_filtered["Lat"].mean() if not df_filtered.empty else df_locations["Lat"].mean()
    center_lon = df_filtered["Long"].mean() if not df_filtered.empty else df_locations["Long"].mean()
    m_all = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    for _, row in df_filtered.iterrows():
        disp_label = row["display_cluster"]
        color = "black" if disp_label == "Noise" else cluster_colors[int(disp_label) - 1]
        folium.CircleMarker(location=[row["Lat"], row["Long"]], radius=4, color=color,
                            fill=True, fill_color=color, fill_opacity=0.8,
                            tooltip=f"{row['location_id']}, Cluster: {disp_label}").add_to(m_all)
    # Add cluster circles and markers using snapped highway coordinates (if available)
    clusters_group = df_locations.groupby("final_cluster")
    for cluster, group in clusters_group:
        display_lbl = cluster + 1
        snapped_center = (group["snapped_lat_highway"].iloc[0], group["snapped_lon_highway"].iloc[0])
        max_dist = group.apply(
            lambda r: haversine_distance(group["Lat"].mean(), group["Long"].mean(), r["Lat"], r["Long"]), axis=1
        ).max()
        radius_m = max_dist * 1000
        color = cluster_colors[cluster % len(cluster_colors)]
        folium.Circle(location=[group["Lat"].mean(), group["Long"].mean()], radius=radius_m, color=color,
                      fill=False, popup=f"Cluster {display_lbl} Radius: {radius_m:.0f} m").add_to(m_all)
        folium.Marker(location=snapped_center, popup=f"Cluster {display_lbl} Center",
                      icon=folium.Icon(color=color, icon="info-sign")).add_to(m_all)
    legend_html = """
    <div style='position: fixed; bottom: 50px; left: 50px; width: 220px; height: 120px;
         background-color: white; z-index:9999; font-size:14px; border:2px solid grey; padding:10px;'>
      <b>Legend</b><br>
      <i style='background:blue; color:blue;'>&nbsp;&nbsp;__&nbsp;&nbsp;</i> Cluster Colors<br>
      <i style='background:black; color:black;'>&nbsp;&nbsp;__&nbsp;&nbsp;</i> Noise/Unclustered
    </div>
    """
    m_all.get_root().html.add_child(folium.Element(legend_html))
    st_data = st_folium(m_all, width=1200, height=800)
else:
    center_lat = df_filtered["Lat"].mean() if not df_filtered.empty else df_locations["Lat"].mean()
    center_lon = df_filtered["Long"].mean() if not df_filtered.empty else df_locations["Long"].mean()
    m_single = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    for _, row in df_filtered.iterrows():
        disp_label = row["display_cluster"]
        color = "black" if disp_label == "Noise" else cluster_colors[int(disp_label) - 1]
        folium.CircleMarker(location=[row["Lat"], row["Long"]], radius=4, color=color,
                            fill=True, fill_color=color, fill_opacity=0.8,
                            tooltip=f"{row['location_id']}, Cluster: {disp_label}").add_to(m_single)
    m_single.get_root().html.add_child(folium.Element(legend_html))
    st_data = st_folium(m_single, width=1200, height=800)

st.sidebar.markdown(f"**Cluster assignments file saved to:** `{OUTPUT_FILE}`")
logging.info("Streamlit app loaded.")