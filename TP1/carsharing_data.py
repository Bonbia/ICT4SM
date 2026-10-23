"""
carsharing_data.py - chargement et filtrage des bookings / parkings (ICT4SM Lab 1).

Utilisation depuis n'importe quel notebook placé dans le même dossier :

    from carsharing_data import get_filtered_data

    bookings_f, parkings_f = get_filtered_data()

    # avec les données brutes et les bornes de zone gardées :
    bookings_f, parkings_f, extra = get_filtered_data(return_extra=True)
    extra["raw"]["bookings"]["Torino"]     # DataFrame non filtré
    extra["zones"][("bookings", "Torino")] # (a, b) en minutes
"""
from datetime import datetime, timezone
import os
import pickle

import numpy as np
import pandas as pd
import pymongo as pm

# ----------------------------------------------------------------------------
# Paramètres par défaut (identiques à CleanNB)
# ----------------------------------------------------------------------------
CITIES = ["Torino", "Columbus", "Frankfurt"]
START = int(datetime(2017, 11, 1, tzinfo=timezone.utc).timestamp())
END = int(datetime(2018, 2, 1, tzinfo=timezone.utc).timestamp())
Q = 0.90  # fraction de la ECDF conservée

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_db = None


def get_db():
    """Connexion MongoDB (créée une seule fois, à la première utilisation)."""
    global _db
    if _db is None:
        client = pm.MongoClient(
            "bigdatadb.polito.it",
            ssl=True,
            authSource="carsharing",
            username="ictts",
            password="Ict4SM22!",
            tlsAllowInvalidCertificates=True,
        )
        _db = client["carsharing"]
    return _db


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load_wtime(coll, city, start=START, end=END, db=None):
    """Charge une collection pour une ville, avec durée et champs temporels
    (day, week, iso_year, dow, hour) ; duration en secondes, duration_min en minutes."""
    db = db if db is not None else get_db()
    pipe = [
        {"$match": {"city": city, "init_time": {"$gte": start, "$lt": end},"public_transport.duration":{"$}},
        {"$project": {
            "_id": 0,
            "init_time": 1,
            "final_time":1,
            "duration": {"$subtract": ["$final_time", "$init_time"]},
            "day": {"$dateToString": {"date": "$init_date", "format": "%Y-%m-%d"}},
            "week": {"$isoWeek": "$init_date"},
            "iso_year": {"$isoWeekYear": "$init_date"},
            "dow": {"$isoDayOfWeek": "$init_date"},
            "hour": {"$hour": "$init_date"},
            "vendor": 1,
            "init_address": 1,
            "final_address": 1,
            "origin_destination": 1,
            "public_transport":1
   
        }},
    ]
    df = pd.DataFrame(db[coll].aggregate(pipe, allowDiskUse=True))
    df["duration_min"] = df["duration"] / 60
    return df


# ----------------------------------------------------------------------------
# Filtrage : zone la plus dense de la ECDF
# ----------------------------------------------------------------------------
def best_zone(x, q=Q, log=True):
    """Plus petit intervalle [a, b] contenant une fraction q des échantillons.
    log=True : largeur mesurée en log(durée) ; les durées <= 0 sont ignorées."""
    x = np.sort(np.asarray(x, dtype=float))
    if log:
        x = x[x > 0]
    t = np.log(x) if log else x
    n = len(x)
    k = int(np.ceil(q * n))
    widths = t[k - 1:] - t[:n - k + 1]
    i = np.argmin(widths)
    return x[i], x[i + k - 1]


def keep_best_zone(df, q=Q, col="duration_min", log=True):
    """Garde les lignes dont la durée est dans la zone la plus dense.
    Retourne (df_filtré, (a, b))."""
    a, b = best_zone(df[col].values, q, log)
    return df[df[col].between(a, b)].copy(), (a, b)


# ----------------------------------------------------------------------------
# Fonction principale
# ----------------------------------------------------------------------------
def get_filtered_data(cities=None, start=START, end=END, q=Q,
                      use_cache=True, refresh=False, verbose=True,
                      return_extra=False):
    """
    Retourne (bookings_f, parkings_f) : deux dicts {ville: DataFrame} filtrés
    (zone la plus dense contenant la fraction `q` de la ECDF des durées).

    Paramètres
    ----------
    cities       : liste de villes (défaut : Torino, Columbus, Frankfurt)
    start, end   : bornes en timestamp UNIX (défaut : 01/11/2017 - 01/02/2018)
    q            : fraction de la ECDF conservée (défaut 0.90)
    use_cache    : sauvegarde / relit les données brutes dans TP1/.cache
                   (évite de réinterroger MongoDB à chaque notebook)
    refresh      : True = ignore le cache et recharge depuis MongoDB
    verbose      : affiche la zone gardée et le nombre de lignes conservées
    return_extra : True = retourne aussi {"raw": ..., "zones": ...}
    """
    cities = list(cities) if cities else list(CITIES)

    raw = None
    cache_file = os.path.join(_CACHE_DIR, f"raw_{start}_{end}_{'-'.join(cities)}.pkl")
    if use_cache and not refresh and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            raw = pickle.load(f)

    if raw is None:
        raw = {
            "bookings": {c: load_wtime("PermanentBookings", c, start, end) for c in cities},
            "parkings": {c: load_wtime("PermanentParkings", c, start, end) for c in cities},
        }
        if use_cache:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(raw, f)

    filtered = {kind: {} for kind in raw}
    zones = {}
    for kind, data in raw.items():
        for c in cities:
            filtered[kind][c], zones[(kind, c)] = keep_best_zone(data[c], q)
            if verbose:
                a, b = zones[(kind, c)]
                n0, n1 = len(data[c]), len(filtered[kind][c])
                print(f"{kind:9s} {c:10s} zone = [{a:7.1f}, {b:8.1f}] min "
                      f"-> {n1}/{n0} gardés ({n1 / n0:.1%})")

    if return_extra:
        return filtered["bookings"], filtered["parkings"], {"raw": raw, "zones": zones}
    return filtered["bookings"], filtered["parkings"]


# ----------------------------------------------------------------------------
# Vérification de la complétude des données
# ----------------------------------------------------------------------------
def check_data(data, start=START, end=END, low_ratio=0.3, verbose=True):
    """
    Vérifie que les données couvrent bien toute la période [start, end[.

    data : le dict {ville: DataFrame} (bookings ou parkings, brut ou filtré)
           ou un dict {"bookings": {...}, "parkings": {...}} (ex. extra["raw"]).
    low_ratio : un jour est signalé "faible" si son nb de lignes < low_ratio x médiane.

    Pour chaque ville, contrôle :
      - première / dernière date (UTC) vs bornes attendues
      - nb de jours couverts vs nb de jours attendus, liste des jours manquants
      - jours à activité anormalement faible (trou partiel de collecte)
      - timestamps hors période, valeurs nulles, durées <= 0, doublons
    Retourne un DataFrame récapitulatif (colonne "ok" = tout est bon).
    """
    # accepte {"bookings": {...}, "parkings": {...}} ou directement {ville: df}
    if data and all(isinstance(v, dict) for v in data.values()):
        items = [(f"{k}/{c}", df) for k, d in data.items() for c, df in d.items()]
    else:
        items = list(data.items())

    exp_days = pd.date_range(pd.Timestamp(start, unit="s"),
                             pd.Timestamp(end, unit="s"), freq="D", inclusive="left")
    rows = []
    for name, df in items:
        if len(df) == 0:
            rows.append({"jeu": name, "lignes": 0, "ok": False})
            continue
        ts = pd.to_datetime(df["init_time"], unit="s")
        per_day = ts.dt.floor("D").value_counts().reindex(exp_days, fill_value=0)
        missing = [d.strftime("%Y-%m-%d") for d in per_day.index[per_day == 0]]
        med = per_day[per_day > 0].median()
        low = [d.strftime("%Y-%m-%d") for d in per_day.index[(per_day > 0) & (per_day < low_ratio * med)]]
        out = int(((df["init_time"] < start) | (df["init_time"] >= end)).sum())
        nulls = int(df[["init_time", "duration"]].isna().sum().sum())
        neg = int((df["duration"] <= 0).sum())
        dup = int(df.duplicated(subset=["init_time", "duration"]).sum())

        r = {
            "jeu": name,
            "lignes": len(df),
            "première date": ts.min().strftime("%Y-%m-%d %H:%M"),
            "dernière date": ts.max().strftime("%Y-%m-%d %H:%M"),
            "jours couverts": f"{len(exp_days) - len(missing)}/{len(exp_days)}",
            "jours manquants": len(missing),
            "jours faibles": len(low),
            "hors période": out,
            "nulls": nulls,
            "durée <= 0": neg,
            "doublons": dup,
            "ok": not (missing or low or out or nulls),
        }
        rows.append(r)
        if verbose and (missing or low):
            if missing:
                print(f"[{name}] jours SANS données : {missing[:15]}{' ...' if len(missing) > 15 else ''}")
            if low:
                print(f"[{name}] jours à activité très faible (< {low_ratio:.0%} de la médiane) : "
                      f"{low[:15]}{' ...' if len(low) > 15 else ''}")

    summary = pd.DataFrame(rows).set_index("jeu")
    if verbose:
        print(f"Période attendue : {exp_days[0]:%Y-%m-%d} -> {exp_days[-1]:%Y-%m-%d} ({len(exp_days)} jours, UTC)")
        print("TOUT EST OK" if summary["ok"].all() else "ATTENTION : voir les lignes avec ok=False")
    return summary
