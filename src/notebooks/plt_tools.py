"""Collection of utilities functions."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def frodi_leciti_per_regione(df):
    """Esplora distribuzione di sinistri frodi e leciti per regione."""
    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_plt = (
        df_wrk.groupby(["region_of_claim", "tipo_sinistro"])
        .agg({"dummy_index": "sum"})
        .reset_index()
    )
    df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "frode", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
    df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "lecito", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(
        x="region_of_claim", y="dummy_index", hue="tipo_sinistro", data=df_plt
    )
    plt.ylabel("Percentuale", fontsize=18)
    plt.xlabel("Regione", fontsize=18)
    plt.title("Distribuzione di sinistri frodi e leciti per regione", fontsize=26)
    plt.show()


def sinistri_per_regione(df):
    """Esplora distribuzione di sinistri per regione."""
    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_plt = (
        df_wrk.groupby(["region_of_claim"]).agg({"dummy_index": "sum"}).reset_index()
    )
    df_plt["prcn"] = 100 * df_plt["dummy_index"] / len(df_wrk)
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x="region_of_claim", y="dummy_index", data=df_plt)
    plt.ylabel("Numero di sinistri", fontsize=18)
    plt.xlabel("Regione", fontsize=18)
    plt.title("Distribuzione di sinistri per regione", fontsize=26)
    for index, row in df_plt.iterrows():
        plt.text(
            row.name,
            row.dummy_index,
            f"{round(row.prcn,1)} %",
            color="black",
            ha="center",
        )
    plt.show()


def sinistri_polizze_per_regione(df):
    """Distribuzione geografica di polizze stipulate e sinistri accaduti."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_rop = (
        df_wrk.groupby(["region_of_policy"]).agg({"dummy_index": "sum"}).reset_index()
    )
    df_rop.rename(columns={"region_of_policy": "region"}, inplace=True)
    df_rop["obs_type"] = "stipula polizza"
    df_rop["region"] = df_rop["region"].apply(
        lambda x: "other" if x in ["puglia", "liguria", "none"] else x
    )
    df_roc = (
        df_wrk.groupby(["region_of_claim"]).agg({"dummy_index": "sum"}).reset_index()
    )
    df_roc.rename(columns={"region_of_claim": "region"}, inplace=True)
    df_roc["obs_type"] = "accadimento sinistro"
    df_roc["region"] = df_roc["region"].apply(
        lambda x: "other" if x in ["puglia", "liguria", "none"] else x
    )
    df_plt = pd.concat([df_roc, df_rop], axis=0)
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x="region", y="dummy_index", hue="obs_type", data=df_plt)
    plt.ylabel("Numero osservazioni", fontsize=18)
    plt.xlabel("Regione", fontsize=18)
    plt.title(
        "Distribuzione geografica polizze stipulate e sinistri accaduti", fontsize=26
    )
    plt.show()


def coerenza_regione_polizza_sinistro(df):
    """Compara sinitri leciti e frodi quando regione di stipula polizza e accadimento sono uguali o diverse."""
    df_wrk = df.copy()
    df_wrk["check_coerenza_regioni"] = df_wrk.apply(
        lambda row: "regioni coerenti"
        if row.region_of_claim == row.region_of_policy
        else "regioni non coerenti",
        axis=1,
    )
    df_wrk["dummy_index"] = 1
    df_plt = (
        df_wrk.groupby(["check_coerenza_regioni", "tipo_sinistro"])
        .agg({"dummy_index": "sum"})
        .reset_index()
    )
    df_plt["prcn"] = df_plt["dummy_index"]
    df_plt.loc[df_plt["tipo_sinistro"] == "frode", "prcn"] = (
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "prcn"]
        / df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"].sum()
    )
    df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "prcn"] = (
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "prcn"]
        / df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"].sum()
    )

    plt.figure(figsize=(16, 9))
    ax = sns.barplot(
        x="check_coerenza_regioni", y="prcn", hue="tipo_sinistro", data=df_plt
    )
    plt.ylabel("Percentuale", fontsize=18)
    plt.xlabel("Regione", fontsize=18)
    plt.title(
        "Percentuale di sinistri leciti e frodi suddivisi per coerenza tra regione di stipula polizza e avvenuto sinistro",
        fontsize=22,
    )
    plt.show()


def distanza_temporale_accaduto_dichiarato(df, split_frodi=True):
    """Plot distanza temporale in giorni tra sinistro accaduto e dichiarato."""

    plt.figure(figsize=(16, 9))
    if split_frodi == True:
        ax = sns.distplot(
            df.loc[df.tipo_sinistro == "lecito", "diff_days_claim_date_notif_date"],
            rug=False,
            hist=True,
        )
        ax = sns.distplot(
            df.loc[df.tipo_sinistro == "frode", "diff_days_claim_date_notif_date"],
            rug=False,
            hist=True,
        )
        plt.legend(["sinistri leciti", "frodi"])
    else:
        ax = sns.distplot(
            df.loc[:, "diff_days_claim_date_notif_date"], rug=False, hist=True,
        )
    plt.ylabel("", fontsize=18)
    plt.xlabel("Giorni trascorsi", fontsize=18)
    plt.xlim(0, 200)
    plt.title("Distanza temporale tra sinistro accaduto e dichiarato", fontsize=26)
    plt.show()


def distanza_temporale_apertura_sinistro(df, split_frodi=True):
    """Plot distanza temporale in giorni tra sinistro accaduto e apertura polizza."""

    df = df.loc[df.diff_days_claim_date_policy_start_date > 0]
    plt.figure(figsize=(16, 9))
    if split_frodi == True:
        ax = sns.distplot(
            df.loc[
                df.tipo_sinistro == "lecito", "diff_days_claim_date_policy_start_date"
            ],
            rug=False,
            hist=True,
        )
        ax = sns.distplot(
            df.loc[
                df.tipo_sinistro == "frode", "diff_days_claim_date_policy_start_date"
            ],
            rug=False,
            hist=True,
        )
        plt.legend(["sinistri leciti", "frodi"])
    else:
        ax = sns.distplot(
            df.loc[:, "diff_days_claim_date_policy_start_date"], rug=False, hist=True,
        )
    plt.ylabel("", fontsize=18)
    plt.xlabel("Giorni trascorsi", fontsize=18)
    plt.xlim(0, 1500)
    plt.title(
        "Distanza temporale tra apertura polizza e avvenuto sinistro", fontsize=26
    )
    plt.show()


def distanza_temporale_chiusura_sinistro(df, split_frodi=True):
    """Plot distanza temporale in giorni tra sinistro accaduto e apertura polizza."""

    df = df.loc[df.diff_days_claim_date_policy_end_date > 0]
    plt.figure(figsize=(16, 9))
    if split_frodi == True:
        ax = sns.distplot(
            df.loc[
                df.tipo_sinistro == "lecito", "diff_days_claim_date_policy_end_date"
            ],
            rug=False,
            hist=True,
        )
        ax = sns.distplot(
            df.loc[df.tipo_sinistro == "frode", "diff_days_claim_date_policy_end_date"],
            rug=False,
            hist=True,
        )
        plt.legend(["sinistri leciti", "frodi"])
    else:
        ax = sns.distplot(
            df.loc[:, "diff_days_claim_date_policy_end_date"], rug=False, hist=True,
        )
    plt.ylabel("", fontsize=18)
    plt.xlabel("Giorni trascorsi", fontsize=18)
    plt.xlim(0, 1500)
    plt.title(
        "Distanza temporale tra avvenuto sinistro e scadenza polizza", fontsize=26
    )
    plt.show()


def eta_assicurato_controparte(df):
    """Plot distribuzione di età assicurato e controparte."""

    plt.figure(figsize=(16, 9))
    ax = sns.distplot(
        df.diff_year_now_fp__date_of_birth, rug=False, hist=True, kde=False, bins=20
    )
    ax = sns.distplot(
        df.diff_year_now_tp__date_of_birth, rug=False, hist=True, kde=False, bins=20
    )
    plt.ylabel("", fontsize=18)
    plt.xlabel("Età", fontsize=18)
    plt.xlim(0, 100)
    plt.title("Età di assicurato e controparte", fontsize=26)
    plt.legend(["Assicurato", "Controparte"])
    plt.show()


def differenza_eta_assicurato_controparte(df, split_frodi=True):
    """Plot differenza di eta tra assicurato e controparte divisi per frodi / leciti."""

    df["differenza_eta"] = (
        df["diff_year_now_fp__date_of_birth"] - df["diff_year_now_tp__date_of_birth"]
    ).abs()
    plt.figure(figsize=(16, 9))
    if split_frodi == True:
        ax = sns.distplot(
            df.loc[df.tipo_sinistro == "lecito", "differenza_eta"],
            rug=False,
            hist=True,
            kde=True,
            bins=30,
        )
        ax = sns.distplot(
            df.loc[df.tipo_sinistro == "frode", "differenza_eta"],
            rug=False,
            hist=True,
            kde=True,
            bins=30,
        )
        plt.legend(["Frode", "Lecito"])
    else:
        ax = sns.distplot(
            df.loc[:, "differenza_eta"], rug=False, hist=True, kde=True, bins=30,
        )
    plt.ylabel("", fontsize=18)
    plt.xlabel("Età", fontsize=18)
    plt.xlim(0, 50)
    plt.title("Differenza di età tra assicurato e controparte", fontsize=26)
    plt.show()


def presenza_testimoni(df):
    """Esplora presenza di testimoni."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_wrk.is_witness = df_wrk.is_witness.apply(
        lambda x: "Testimoni presenti" if x == True else "Testimoni assenti"
    )
    df_plt = df_wrk.groupby(["is_witness"]).agg({"dummy_index": "sum"}).reset_index()
    df_plt["prcn"] = 100 * df_plt["dummy_index"] / len(df_wrk)
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x="is_witness", y="dummy_index", data=df_plt)
    plt.ylabel("Numero di sinistri", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.title("Presenza di testimoni", fontsize=26)
    for index, row in df_plt.iterrows():
        plt.text(
            row.name,
            row.dummy_index,
            f"{round(row.prcn,1)} %",
            color="black",
            ha="center",
        )
    ax.tick_params(axis="both", which="major", labelsize=18)
    plt.show()


def presenza_testimoni_tipo_sinistro(df):
    """Espora testimoni per tipo sinistro."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_wrk.is_witness = df_wrk.is_witness.apply(
        lambda x: "Testimoni presenti" if x == True else "Testimoni assenti"
    )
    df_plt = (
        df_wrk.groupby(["is_witness", "tipo_sinistro"])
        .agg({"dummy_index": "sum"})
        .reset_index()
    )
    df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "frode", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
    df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "lecito", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x="tipo_sinistro", y="dummy_index", hue="is_witness", data=df_plt)
    plt.ylabel("Percentuale di sinistri", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.title("Presenza di testimoni per tipo sinistro", fontsize=26)
    ax.tick_params(axis="both", which="major", labelsize=18)
    plt.show()


def coinvolgimento_furto(df):
    """Espora tipo sinistro per coinvolgimento in furto."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_wrk.is_thief_known = df_wrk.is_thief_known.apply(
        lambda x: "Coinvolto in furto" if x == True else "Pulito"
    )
    df_plt = (
        df_wrk.groupby(["is_thief_known", "tipo_sinistro"])
        .agg({"dummy_index": "sum"})
        .reset_index()
    )
    df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "frode", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
    df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "lecito", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(
        x="tipo_sinistro", y="dummy_index", hue="is_thief_known", data=df_plt
    )
    plt.ylabel("Percentuale di sinistri", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.title("Veicolo coinvolto in furto per tipo di sinistro", fontsize=26)
    ax.tick_params(axis="both", which="major", labelsize=18)
    plt.show()


def distanza_contraente_terzaparte(df, ssg, ssf):
    """Esplora la distanza per contraente e terza parte."""

    sample_size_good = ssg
    sct_data_good = (
        df.loc[df.tipo_sinistro == "lecito"]
        .sample(
            min(sample_size_good, len(df.loc[df.tipo_sinistro == "lecito"])),
            random_state=42,
        )
        .copy()
    )
    sample_size_bad = ssf
    sct_data_bad = (
        df.loc[df.tipo_sinistro == "frode"]
        .sample(
            min(sample_size_bad, len(df.loc[df.tipo_sinistro == "frode"])),
            random_state=42,
        )
        .copy()
    )
    sct_data = pd.concat([sct_data_good, sct_data_bad]).sample(frac=1)

    plt.figure(figsize=(16, 9))
    ax = sns.scatterplot(
        x="dist_claim_tp",
        y="dist_claim_fp",
        data=sct_data.loc[sct_data.tipo_sinistro == "lecito"],
        s=45,
        alpha=0.8,
    )
    ax = sns.scatterplot(
        x="dist_claim_tp",
        y="dist_claim_fp",
        data=sct_data.loc[sct_data.tipo_sinistro == "frode"],
        s=45,
        alpha=0.8,
    )
    plt.legend(title="", loc="right", labels=["Sinistri leciti", "Frodi"], fontsize=16)
    plt.grid(color="grey", linestyle="--", linewidth=0.5, which="both")
    plt.ylabel("Distanza terza parte", fontsize=18)
    plt.xlabel("Distanza contraente polizza", fontsize=18)
    plt.title(
        "Distanza tra luogo di sottoscrizione polizza e luogo di accadimento sinistro",
        fontsize=26,
    )
    plt.show()


def distanza_contraente_terzaparte_generale(df, s_size):
    """Esplora la distanza per contraente e terza parte."""

    df_wrk = df.sample(min(s_size, len(df))).copy()

    plt.figure(figsize=(16, 9))
    ax = sns.scatterplot(
        x="dist_claim_tp", y="dist_claim_fp", data=df_wrk, s=45, alpha=0.8,
    )
    plt.grid(color="grey", linestyle="--", linewidth=0.5, which="both")
    plt.ylabel("Distanza terza parte", fontsize=18)
    plt.xlabel("Distanza contraente polizza", fontsize=18)
    plt.title(
        "Distanza tra luogo di sottoscrizione polizza e luogo di accadimento sinistro",
        fontsize=26,
    )
    plt.show()


def compilazione_cid(df, split_frodi=True):
    """Espora la compilazione di costatazione amichevole per frodi e sinistri leciti."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    if split_frodi == True:
        df_plt = (
            df_wrk.groupby(["flag_cid_compiled", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="tipo_sinistro", y="dummy_index", hue="flag_cid_compiled", data=df_plt
        )
        plt.ylabel("Percentuale di sinistri", fontsize=18)
        plt.xlabel("", fontsize=18)
        plt.title("Compilazione constatazione amichevole", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=18)
        plt.show()
    else:
        df_plt = (
            df_wrk.groupby(["flag_cid_compiled"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="flag_cid_compiled", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("", fontsize=18)
        plt.title("Compilazione constatazione amichevole", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=18)
        plt.show()


def intervento_polizia(df, split_frodi=True):
    """La polizia è intervenuta o meno."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_wrk.is_police_report = df_wrk.is_police_report.apply(
        lambda x: "Polizia intervenuta" if x == True else "Polizia non intervenuta"
    )
    if split_frodi == True:
        df_plt = (
            df_wrk.groupby(["is_police_report", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="tipo_sinistro", y="dummy_index", hue="is_police_report", data=df_plt
        )
        plt.ylabel("Percentuale di sinistri", fontsize=18)
        plt.xlabel("", fontsize=18)
        plt.title("Intervento polizia per tipo sinistro", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=18)
        plt.show()
    else:
        df_plt = (
            df_wrk.groupby(["is_police_report"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="is_police_report", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("", fontsize=18)
        plt.title("Intervento polizia per tipo sinistro", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=18)
        plt.show()


def controparte_assicurata(df):
    """La controparte era assicurata oppure no."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    df_wrk.tp__thirdparty_is_insured = df_wrk.tp__thirdparty_is_insured.apply(
        lambda x: "Controparte assicurata"
        if x == True
        else "Controparte non assicurata"
    )
    df_plt = (
        df_wrk.groupby(["tp__thirdparty_is_insured", "tipo_sinistro"])
        .agg({"dummy_index": "sum"})
        .reset_index()
    )
    df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "frode", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
    df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
        df_plt["tipo_sinistro"] == "lecito", "dummy_index"
    ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(
        x="tipo_sinistro", y="dummy_index", hue="tp__thirdparty_is_insured", data=df_plt
    )
    plt.ylabel("Percentuale di sinistri", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.title("Controparte assicurata", fontsize=26)
    ax.tick_params(axis="both", which="major", labelsize=18)
    plt.show()


def frodi_leciti_per_claim(df):
    """Esplora distribuzione di sinistri frodi e leciti per tipologia claim."""

    i = 1
    for var in ["claim_description", "claim_desc", "claim_desc2"]:

        df_wrk = df.copy()
        df_wrk["dummy_index"] = 1
        df_plt = (
            df_wrk.groupby([var, "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x=var, y="dummy_index", hue="tipo_sinistro", data=df_plt)
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Descrizione del sinistro", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione di sinistri frodi e leciti per descrizione accadimento {i}/3",
            fontsize=26,
        )
        plt.show()
        i = i + 1


def sinistri_per_claim(df):
    """Esplora distribuzione di sinistri per tipologia claim."""

    i = 1
    for var in ["claim_description", "claim_desc", "claim_desc2"]:

        df_wrk = df.copy()
        df_wrk["dummy_index"] = 1
        df_plt = df_wrk.groupby([var]).agg({"dummy_index": "sum"}).reset_index()
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x=var, y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("Descrizione del sinistro", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione di sinistri per descrizione accadimento {i}/3", fontsize=26,
        )
        plt.show()
        i = i + 1


def tipologia_di_tariffa(df, split_frodi=True):
    """Sinistri frodi e leciti suddivisi per tipologia di tariffa."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    if split_frodi == True:
        df_plt = (
            df_wrk.groupby(["tarif_type", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="tarif_type", y="dummy_index", hue="tipo_sinistro", data=df_plt
        )
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Tipologia di tariffa", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione di sinistri frodi e leciti per tipologia di tariffa",
            fontsize=26,
        )
        plt.show()
    else:
        df_plt = (
            df_wrk.groupby(["tarif_type"]).agg({"dummy_index": "sum"}).reset_index()
        )
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="tarif_type", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("Tipologia di tariffa", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione di sinistri per tipologia di tariffa", fontsize=26,
        )
        plt.show()


def tipologia_di_patente(df, split_frodi=True):
    """Sinistri frodi e leciti separati per tipologia di patente."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1
    if split_frodi == True:
        df_plt = (
            df_wrk.groupby(["driving_licence_type", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="driving_licence_type", y="dummy_index", hue="tipo_sinistro", data=df_plt
        )
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Tipologia di patente", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione di sinistri frodi e leciti per tipologia di patente",
            fontsize=26,
        )
        plt.show()
    else:
        df_plt = (
            df_wrk.groupby(["driving_licence_type"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="driving_licence_type", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("Tipologia di patente", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione di sinistri per tipologia di patente", fontsize=26,
        )
        plt.show()


def marca_veicolo_generale(df, omit_noise, split_frodi=True):
    """Esplora distribuzione dei sinistri per marca di veicolo coinvolto."""

    if "tipo_sinistro" not in df:
        df["tipo_sinistro"] = "1"

    df_wrk_fp = df.copy()
    df_wrk_fp["marca"] = df_wrk_fp.fp__vehicle_make
    df_wrk_fp["soggetto"] = "contraente"
    df_wrk_fp = df_wrk_fp[["claim_id", "tipo_sinistro", "marca", "soggetto"]]
    df_wrk_tp = df.copy()
    df_wrk_tp["marca"] = df_wrk_tp.tp__vehicle_make
    df_wrk_tp["soggetto"] = "terza parte"
    df_wrk_tp = df_wrk_tp[["claim_id", "tipo_sinistro", "marca", "soggetto"]]
    df_wrk = pd.concat([df_wrk_tp, df_wrk_fp])
    del df_wrk_tp, df_wrk_fp
    df_wrk["dummy_index"] = 1
    if omit_noise:
        df_wrk = df_wrk.loc[df_wrk.marca.isin(["none", "other"]) == False].copy()

    # Plot senza distinzione per tipologia di sinistro
    df_plt = df_wrk.groupby(["marca"]).agg({"dummy_index": "sum"}).reset_index()
    df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x="marca", y="dummy_index", data=df_plt)
    plt.ylabel("Numero di sinistri", fontsize=18)
    plt.xlabel("Marca dei veicoli coinvolti", fontsize=18)
    plt.xticks(rotation=30)
    plt.title(
        f"Principali marche coinvolte in sinistri", fontsize=26,
    )
    plt.show()

    # Plot distinguendo per tipologia di sinistro
    if split_frodi == True:
        df_plt = (
            df_wrk.groupby(["marca", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="marca", y="dummy_index", hue="tipo_sinistro", data=df_plt)
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Marca dei veicoli coinvolti", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione marche per sinistri fraudolenti e leciti", fontsize=26,
        )
        plt.show()


def tipologia_veicolo_generale(df, omit_noise, split_frodi=True):
    """Esplora distribuzione dei sinistri per marca di veicolo coinvolto."""

    df_wrk_fp = df.copy()
    df_wrk_fp["tipologia"] = df_wrk_fp.fp__vehicle_type
    df_wrk_fp["soggetto"] = "contraente"
    df_wrk_fp = df_wrk_fp[["claim_id", "tipo_sinistro", "tipologia", "soggetto"]]
    df_wrk_tp = df.copy()
    df_wrk_tp["tipologia"] = df_wrk_tp.tp__vehicle_type
    df_wrk_tp["soggetto"] = "terza parte"
    df_wrk_tp = df_wrk_tp[["claim_id", "tipo_sinistro", "tipologia", "soggetto"]]
    df_wrk = pd.concat([df_wrk_tp, df_wrk_fp])
    del df_wrk_tp, df_wrk_fp
    df_wrk["dummy_index"] = 1
    if omit_noise:
        df_wrk = df_wrk.loc[df_wrk.marca.isin(["none", "other"]) == False].copy()

    # Plot senza distinzione per tipologia di sinistro
    df_plt = df_wrk.groupby(["tipologia"]).agg({"dummy_index": "sum"}).reset_index()
    df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x="tipologia", y="dummy_index", data=df_plt)
    plt.ylabel("Numero di sinistri", fontsize=18)
    plt.xlabel("Tipologia dei veicoli coinvolti", fontsize=18)
    plt.xticks(rotation=30)
    plt.title(
        f"Tipologia veicoli coinvolti in sinistri", fontsize=26,
    )
    plt.show()

    # Plot distinguendo per tipologia di sinistro
    if split_frodi == True:
        df_plt = (
            df_wrk.groupby(["tipologia", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["dummy_index"], ascending=False, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="tipologia", y="dummy_index", hue="tipo_sinistro", data=df_plt
        )
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Tipologia dei veicoli coinvolti", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Distribuzione tipologia veicolo per sinistri fraudolenti e leciti",
            fontsize=26,
        )
        plt.show()


def complessita_sinistro(df, split_frodi=False):
    """Esplora complessità del sinistro."""

    df_wrk = df.copy()
    df_wrk["complessita_sinistro"] = df_wrk["complessita_sinistro"].astype(str)
    df_wrk["dummy_index"] = 1

    if split_frodi == False:
        # Plot senza distinzione per tipologia di sinistro
        df_plt = (
            df_wrk.groupby(["complessita_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.sort_values(by=["complessita_sinistro"], ascending=True, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="complessita_sinistro", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("Complessità sinistro", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Sinistri suddivisi per indice di complessità.", fontsize=26,
        )
        plt.show()

    if split_frodi == True:
        # Plot distinguendo per tipologia di sinistro
        df_plt = (
            df_wrk.groupby(["complessita_sinistro", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["complessita_sinistro"], ascending=True, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="complessita_sinistro", y="dummy_index", hue="tipo_sinistro", data=df_plt
        )
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Complessità sinistro", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Sinistri suddivisi per indice di complessità, fraudolenti e leciti",
            fontsize=26,
        )
        plt.show()


def amount_category(df, split_frodi=False):
    """Esplora categoria di costo del sinistro."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1

    if split_frodi == False:
        # Plot senza distinzione per tipologia di sinistro
        df_plt = (
            df_wrk.groupby(["claim_amount_category"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.sort_values(by=["claim_amount_category"], ascending=True, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="claim_amount_category", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("Categoria di entità danno", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Sinistri suddivisi per categoria di entità danno.", fontsize=26,
        )
        plt.show()

    if split_frodi == True:
        # Plot distinguendo per tipologia di sinistro
        df_plt = (
            df_wrk.groupby(["claim_amount_category", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["claim_amount_category"], ascending=True, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="claim_amount_category", y="dummy_index", hue="tipo_sinistro", data=df_plt
        )
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Categoria entità danno", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Sinistri suddivisi per categoria di entità danno, fraudolenti e leciti",
            fontsize=26,
        )
        plt.show()


def reserved_category(df, split_frodi=False):
    """Esplora categoria di importo riservato per sinistro."""

    df_wrk = df.copy()
    df_wrk["dummy_index"] = 1

    if split_frodi == False:
        # Plot senza distinzione per tipologia di sinistro
        df_plt = (
            df_wrk.groupby(["total_reserved_category"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.sort_values(by=["total_reserved_category"], ascending=True, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(x="total_reserved_category", y="dummy_index", data=df_plt)
        plt.ylabel("Numero di sinistri", fontsize=18)
        plt.xlabel("Categoria di importo riservato", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Sinistri suddivisi per categoria di importo riservato.", fontsize=26,
        )
        plt.show()

    if split_frodi == True:
        # Plot distinguendo per tipologia di sinistro
        df_plt = (
            df_wrk.groupby(["total_reserved_category", "tipo_sinistro"])
            .agg({"dummy_index": "sum"})
            .reset_index()
        )
        df_plt.loc[df_plt["tipo_sinistro"] == "frode", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "frode", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "frode"])
        df_plt.loc[df_plt["tipo_sinistro"] == "lecito", "dummy_index"] = df_plt.loc[
            df_plt["tipo_sinistro"] == "lecito", "dummy_index"
        ] / len(df_wrk.loc[df_wrk["tipo_sinistro"] == "lecito"])
        df_plt.sort_values(by=["total_reserved_category"], ascending=True, inplace=True)
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(
            x="total_reserved_category",
            y="dummy_index",
            hue="tipo_sinistro",
            data=df_plt,
        )
        plt.ylabel("Percentuale", fontsize=18)
        plt.xlabel("Categoria importo riservato", fontsize=18)
        plt.xticks(rotation=30)
        plt.title(
            f"Sinistri suddivisi per categoria di importo riservato, fraudolenti e leciti",
            fontsize=26,
        )
        plt.show()
