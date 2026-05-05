# Databricks notebook source
# MAGIC %md
# MAGIC # Analyse Exploratoire des Données — Steam (Ubisoft)
# MAGIC
# MAGIC **Contexte** : Ubisoft souhaite comprendre l'écosystème du jeu vidéo sur Steam  
# MAGIC **Dataset** : `s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json`  
# MAGIC **Stack** : PySpark + Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Chargement du dataset

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Initialisation Spark (automatique dans Databricks, mais on s'assure d'avoir une session)
spark = SparkSession.builder.getOrCreate()

# Chargement du JSON semi-structuré depuis S3
raw_df = spark.read.option("multiLine", True).json(
    "s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json"
)

print(f"Nombre de lignes : {raw_df.count()}")
print(f"Nombre de colonnes : {len(raw_df.columns)}")
raw_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Exploration du schema & nettoyage de base

# COMMAND ----------

# Aperçu global
raw_df.display()

# COMMAND ----------

# Colonnes disponibles
raw_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Construction du DataFrame principal (df_games)

# COMMAND ----------

# DBTITLE 1,Cell 8
df_games = raw_df.select(
    F.col("data.appid").cast("int").alias("app_id"),
    F.col("data.name").alias("game_name"),
    F.col("data.developer").alias("developer"),
    F.col("data.publisher").alias("publisher"),
    F.expr("try_cast(data.required_age as int)").alias("required_age"),

    # Prix
    F.expr("try_cast(data.price as double)").alias("price_cents"),
    (F.expr("try_cast(data.price as double)") / 100).alias("price_eur"),
    F.expr("try_cast(data.discount as int)").alias("discount_pct"),
    F.when(
        (F.col("data.price").isNull()) | (F.expr("try_cast(data.price as double)") == 0),
        True
    ).otherwise(False).alias("is_free"),

    # Date de sortie
    F.col("data.release_date").alias("release_date_str"),

    # Notes
    F.col("data.positive").cast("int").alias("positive_ratings"),
    F.col("data.negative").cast("int").alias("negative_ratings"),

    # Plateformes
    F.col("data.platforms.windows").cast("boolean").alias("on_windows"),
    F.col("data.platforms.mac").cast("boolean").alias("on_mac"),
    F.col("data.platforms.linux").cast("boolean").alias("on_linux"),

    # Genres et catégories
    F.col("data.genre").alias("genre"),
    F.col("data.categories").alias("categories"),
    F.col("data.languages").alias("supported_languages"),
)

# Nettoyage : suppression des lignes sans nom de jeu
df_games = df_games.filter(F.col("game_name").isNotNull())

# Ratio positif / negatif
df_games = df_games.withColumn(
    "total_ratings",
    F.col("positive_ratings") + F.col("negative_ratings")
).withColumn(
    "positive_ratio",
    F.when(
        F.col("total_ratings") > 0,
        F.round(F.col("positive_ratings") / F.col("total_ratings") * 100, 2)
    ).otherwise(None)
)

# Parsing de la date de sortie -> année (utilise try_to_date pour gérer les formats invalides)
df_games = df_games.withColumn(
    "release_year",
    F.year(F.expr("try_to_date(release_date_str, 'yyyy/MM/d')")).cast("int")
)

print(f"df_games : {df_games.count()} lignes")
df_games.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ANALYSE MACRO

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Editeurs avec le plus de jeux

# COMMAND ----------

df_publisher_count = (
    df_games
    .filter(F.col("publisher").isNotNull())
    .groupBy("publisher")
    .agg(F.count("*").alias("nb_jeux"))
    .orderBy(F.desc("nb_jeux"))
)

df_publisher_count.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Jeux les mieux notes (Metacritic + ratio positif)

# COMMAND ----------

# Top 20 par ratio avis positifs (avec au moins 100 ratings)
df_top_ratio = (
    df_games
    .filter(F.col("total_ratings") >= 100)
    .select("game_name", "publisher", "positive_ratio", "total_ratings")
    .orderBy(F.desc("positive_ratio"))
    .limit(20)
)
df_top_ratio.display()

# COMMAND ----------

# Top 20 par ratio avis positifs (avec au moins 500 ratings)
df_top_ratio = (
    df_games
    .filter(F.col("total_ratings") >= 500)
    .select("game_name", "publisher", "positive_ratio", "total_ratings")
    .orderBy(F.desc("positive_ratio"))
    .limit(20)
)
df_top_ratio.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Sorties de jeux par année (impact COVID ?)

# COMMAND ----------

df_by_year = (
    df_games
    .filter(
        (F.col("release_year").isNotNull()) &
        (F.col("release_year") >= 2000)
    )
    .groupBy("release_year")
    .agg(F.count("*").alias("nb_sorties"))
    .orderBy("release_year")
)

df_by_year.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Distribution des prix

# COMMAND ----------

# Statistiques descriptives
df_games.filter(F.col("is_free") == False).select("price_eur").describe().display()

# COMMAND ----------

# Tranches de prix
df_price_buckets = (
    df_games
    .filter(F.col("is_free") == False)
    .filter(F.col("price_eur").isNotNull())
    .withColumn(
        "price_range",
        F.when(F.col("price_eur") == 0, "Gratuit")
         .when(F.col("price_eur") < 5,  "< 5 EUR")
         .when(F.col("price_eur") < 10, "5 - 10 EUR")
         .when(F.col("price_eur") < 20, "10 - 20 EUR")
         .when(F.col("price_eur") < 40, "20 - 40 EUR")
         .when(F.col("price_eur") < 60, "40 - 60 EUR")
         .otherwise("60+ EUR")
    )
    .groupBy("price_range")
    .agg(F.count("*").alias("nb_jeux"))
    .orderBy("price_range")
)
df_price_buckets.display()

# COMMAND ----------

# Proportion de jeux avec une reduction
df_discount_summary = (
    df_games
    .agg(
        F.count("*").alias("total_jeux"),
        F.count(F.when(F.col("discount_pct") > 0, True)).alias("avec_reduction"),
        F.count(F.when(F.col("is_free") == True, True)).alias("gratuits"),
    )
)
df_discount_summary.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Langues les plus représentées

# COMMAND ----------

# La colonne supported_languages est souvent une string de type "English, French, ..."
# On split et on explode
df_languages = (
    df_games
    .filter(F.col("supported_languages").isNotNull())
    .withColumn(
        "lang_list",
        F.split(
            F.regexp_replace(F.col("supported_languages"), "<[^>]+>", ""),  # retire les balises HTML
            ","
        )
    )
    .withColumn("language", F.explode(F.col("lang_list")))
    .withColumn("language", F.trim(F.col("language")))
    .filter(F.col("language") != "")
    .groupBy("language")
    .agg(F.count("*").alias("nb_jeux"))
    .orderBy(F.desc("nb_jeux"))
    .limit(25)
)
df_languages.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Jeux interdits aux moins de 16 / 18 ans

# COMMAND ----------

df_age_rating = (
    df_games
    .withColumn(
        "age_group",
        F.when(F.col("required_age") >= 18, "18+")
         .when(F.col("required_age") >= 16, "16+")
         .when(F.col("required_age") > 0,   "Autre restriction")
         .otherwise("Tous publics / non renseigne")
    )
    .groupBy("age_group")
    .agg(F.count("*").alias("nb_jeux"))
    .orderBy(F.desc("nb_jeux"))
)
df_age_rating.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ANALYSE DES GENRES

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Construction du DataFrame genres (df_genres)

# COMMAND ----------

# La colonne genre est une string avec des genres séparés par des virgules
# On split et on explode
df_genres = (
    df_games
    .filter(F.col("genre").isNotNull())
    .select(
        "app_id", "game_name", "publisher",
        "positive_ratio", "total_ratings", "price_eur", "is_free",
        "on_windows", "on_mac", "on_linux",
        F.split(F.col("genre"), ", ").alias("genre_list")
    )
    .withColumn("genre", F.explode(F.col("genre_list")))
    .withColumn("genre", F.trim(F.col("genre")))
    .drop("genre_list")
    .filter(F.col("genre") != "")
)

print(f"df_genres : {df_genres.count()} lignes")
df_genres.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Genres les plus représentés

# COMMAND ----------

df_genre_count = (
    df_genres
    .groupBy("genre")
    .agg(F.countDistinct("app_id").alias("nb_jeux"))
    .orderBy(F.desc("nb_jeux"))
)
df_genre_count.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Genres avec le meilleur ratio avis positifs

# COMMAND ----------

df_genre_ratio = (
    df_genres
    .filter(F.col("total_ratings") >= 50)
    .groupBy("genre")
    .agg(
        F.round(F.avg("positive_ratio"), 2).alias("avg_positive_ratio"),
        F.countDistinct("app_id").alias("nb_jeux"),
        F.round(F.avg("total_ratings"), 0).alias("avg_total_ratings"),
    )
    .filter(F.col("nb_jeux") >= 20)
    .orderBy(F.desc("avg_positive_ratio"))
)
df_genre_ratio.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Genres de prédilection par éditeur (Top 10 éditeurs)

# COMMAND ----------

# On prend les 10 éditeurs avec le plus de jeux
top_publishers_df = (
    df_games
    .filter(F.col("publisher").isNotNull())
    .groupBy("publisher")
    .agg(F.count("*").alias("n"))
    .orderBy(F.desc("n"))
    .limit(10)
)

# Extraire la liste des publishers sans utiliser RDD
top_publishers = [row.publisher for row in top_publishers_df.collect()]

df_publisher_genre = (
    df_genres
    .filter(F.col("publisher").isin(top_publishers))
    .groupBy("publisher", "genre")
    .agg(F.countDistinct("app_id").alias("nb_jeux"))
    .orderBy("publisher", F.desc("nb_jeux"))
)
df_publisher_genre.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Genres les plus lucratifs (proxy : prix moyen x nb de jeux)

# COMMAND ----------

df_genre_lucratif = (
    df_genres
    .filter(
        (F.col("is_free") == False) &
        (F.col("price_eur").isNotNull()) &
        (F.col("price_eur") > 0)
    )
    .groupBy("genre")
    .agg(
        F.round(F.avg("price_eur"), 2).alias("prix_moyen_eur"),
        F.countDistinct("app_id").alias("nb_jeux"),
        F.round(F.sum("price_eur"), 2).alias("revenus_potentiels_eur"),
    )
    .filter(F.col("nb_jeux") >= 10)
    .orderBy(F.desc("revenus_potentiels_eur"))
)
df_genre_lucratif.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ANALYSE DES PLATEFORMES

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Disponibilité Windows / Mac / Linux

# COMMAND ----------

total = df_games.count()

df_platform_summary = df_games.agg(
    F.count(F.when(F.col("on_windows") == True, True)).alias("windows"),
    F.count(F.when(F.col("on_mac")     == True, True)).alias("mac"),
    F.count(F.when(F.col("on_linux")   == True, True)).alias("linux"),
    F.count(F.when(
        (F.col("on_windows") == True) &
        (F.col("on_mac")     == True) &
        (F.col("on_linux")   == True), True
    )).alias("all_platforms"),
    F.lit(total).alias("total_jeux"),
)
df_platform_summary.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Genres disponibles de préférence sur Mac / Linux

# COMMAND ----------

# Taux de disponibilité par genre et plateforme
df_genre_platform = (
    df_genres
    .groupBy("genre")
    .agg(
        F.countDistinct("app_id").alias("nb_jeux"),
        F.round(
            F.count(F.when(F.col("on_windows") == True, True)) /
            F.countDistinct("app_id") * 100, 1
        ).alias("pct_windows"),
        F.round(
            F.count(F.when(F.col("on_mac") == True, True)) /
            F.countDistinct("app_id") * 100, 1
        ).alias("pct_mac"),
        F.round(
            F.count(F.when(F.col("on_linux") == True, True)) /
            F.countDistinct("app_id") * 100, 1
        ).alias("pct_linux"),
    )
    .filter(F.col("nb_jeux") >= 20)
    .orderBy(F.desc("pct_linux"))
)
df_genre_platform.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. SYNTHESE — Insights clés pour Ubisoft

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tableau de bord synthétique

# COMMAND ----------

# KPIs globaux
df_games.agg(
    F.count("*").alias("total_jeux"),
    F.countDistinct("publisher").alias("nb_publishers"),
    F.round(F.avg("price_eur"), 2).alias("prix_moyen_eur"),
    F.round(F.avg("positive_ratio"), 2).alias("ratio_positif_moyen_pct"),
    F.count(F.when(F.col("is_free") == True, True)).alias("nb_jeux_gratuits"),
    F.count(F.when(F.col("on_linux") == True, True)).alias("nb_jeux_linux"),
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes methodologiques
# MAGIC
# MAGIC - **Prix** : les prix sont stockés en centimes dans la colonne `price_overview.final`. Division par 100 pour obtenir des euros.
# MAGIC - **Langues** : champ texte brut avec parfois des balises HTML — nettoyé avec `regexp_replace`.
# MAGIC - **Genres** : array de structs `{id, description}` — extrait avec `explode_outer` pour conserver les jeux sans genre.
# MAGIC - **Ratio positif** : calculé uniquement sur les jeux avec au moins 1 avis (évite les divisions par zéro).
# MAGIC - **Revenus potentiels** : proxy basé sur le prix actuel x nb jeux du genre (pas les ventes réelles).
# MAGIC
# MAGIC > **Dataset** : `s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json`