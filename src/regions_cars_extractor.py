# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:24:19 2020

@author: Marco
"""
import os
import re
import pandas as pd
import datetime

def map_values(row, values_dict):
    """Mapping values with macro regione"""
    return values_dict[row]

def split_it(string):
    """Clean a specific field making it consistent."""
    return re.split("__", string)
    
def adding_macro_region(df,drop_policy=0,drop_claim=0, policy = False):
    """Create a macro region type."""
    start = datetime.datetime.now()
    region_of_claim_columns = df.columns[df.columns.str.contains("region_of_claim")]
    region_of_policy_columns = df.columns[df.columns.str.contains("region_of_policy")]

    dict_region = {}
    dict_region["lombardia"] = "NordOvest"
    dict_region["piemonte"] = "NordOvest"
    dict_region["liguria"] = "NordOvest"

    dict_region["veneto"] = "NordEst"
    dict_region["emiliaromagna"] = "NordEst"

    dict_region["toscana"] = "Centro"
    dict_region["lazio"] = "Centro"

    dict_region["puglia"] = "Mezzogiorno"
    dict_region["campania"] = "Mezzogiorno"
    dict_region["sicilia"] = "Mezzogiorno"  # si può mettere anche in sud e isole

    dict_region["other"] = "Altro"
    dict_region["none"] = "Altro"#in alternativa none

    df['policy_claim_same_region'] = 0
    for i in range(len(df)):
        for item in region_of_claim_columns:
           if df[item].iloc[i]==1:
                region_name = split_it(item)[1]
                if df["region_of_claim__"+region_name].iloc[i]:
                    df.at[i, "policy_claim_same_region"]=1

    df["Macro_region_of_claim"] = "Altro"
    for i in range(len(df)):
        for item in region_of_claim_columns:
            if df[item].iloc[i]==1:
                region_name = split_it(item)[1]
                df.at[i, "Macro_region_of_claim"] = map_values(
                        region_name, dict_region
                        )
    if policy:
        df["Macro_region_of_policy"] = "Altro"
        for i in range(len(df)):
            for item in region_of_policy_columns:
                if df[item].iloc[i]:
                    region_name = split_it(item)[1]
                    df.at[i, "Macro_region_of_policy"] = map_values(
                        region_name, dict_region
                    )

    print("it took in total seconds:", datetime.datetime.now() - start)

    if drop_policy:
        if policy:
            df.drop(region_of_policy_columns,axis=1,inplace=True)
    if drop_claim:
    	df.drop(region_of_claim_columns,axis=1,inplace=True)
        
    return df



def create_KPIs():
	morti_feriti = pd.read_excel(
	    "./data/istat/Istat_morti_feriti_incidenti_stradali.xlsx", skiprows=range(0, 12)
	)  # usecols = "A:Z"))
	veicoli_coinvolti = pd.read_excel(
	    "data/istat/Istat_veicoli_coinvolti_in_incidenti.xlsx", skiprows=range(0, 9)
	)  # usecols = "A:Z"))
	totale_veicoli = pd.read_excel(
	    "data/istat/Istat_veicoli_pubblico_registro_automobilistico.xlsx",
	    skiprows=range(0, 5),
	)  # usecols = "A:Z"))

	morti_feriti = morti_feriti[morti_feriti.Macroregione.notna()]
	veicoli_coinvolti = veicoli_coinvolti[veicoli_coinvolti.Macroregione.notna()]
	totale_veicoli = totale_veicoli[totale_veicoli.Macroregione.notna()]

	morti_feriti.groupby(["Macroregione"]).sum()
	veicoli_coinvolti.groupby(["Macroregione"]).sum()
	totale_veicoli.groupby(["Macroregione"]).sum()


	# proxy della gravita degli incidenti: numero di morti o feriti sul totale del parco auto
	Gravita = (
	    morti_feriti.groupby(["Macroregione"]).sum()["morto_o_ferito_totale"]
	    / totale_veicoli.groupby(["Macroregione"]).sum()["parco_veicolare_totale"]
	)

	# proxy della rischiosità di incidenti: numero di incidenti totali sul totale auto
	Pericolosita = (
	    veicoli_coinvolti.groupby(["Macroregione"]).sum()["incidenti_veicoli_totale"]
	    / totale_veicoli.groupby(["Macroregione"]).sum()["parco_veicolare_totale"]
	)

	# proxy della dannosita media rispetto al numero di incidenti: numero di morti su totale incidenti
	Dannosita = (
	    morti_feriti.groupby(["Macroregione"]).sum()["morto_o_ferito_totale"]
	    / veicoli_coinvolti.groupby(["Macroregione"]).sum()["incidenti_veicoli_totale"]
	)

	istat_kpis = pd.concat([Pericolosita,Gravita, Dannosita],axis=1, join_axes=[Dannosita.index]) 
	istat_kpis.columns =["Pericolosita","Gravita","Dannosita"]
    
	return istat_kpis


def add_istat_kpis(df):
    ## adding only from region of claim ##
    istat_kpis = create_KPIs()
    values_dict_P = istat_kpis["Pericolosita"].to_dict()
    values_dict_P['Altro'] = istat_kpis["Pericolosita"].mean()

    df['Macro_region_of_claim_pericolosita'] = df['Macro_region_of_claim'].apply(
            map_values, args = (values_dict_P,))

    values_dict_G = istat_kpis["Gravita"].to_dict()
    values_dict_G['Altro'] = istat_kpis["Gravita"].mean()
    df['Macro_region_of_claim_gravita'] = df['Macro_region_of_claim'].apply(
            map_values, args = (values_dict_G,))

    values_dict_M = istat_kpis["Dannosita"].to_dict()
    values_dict_M['Altro'] = istat_kpis["Dannosita"].mean()
    df['Macro_region_of_claim_dannosita'] = df['Macro_region_of_claim'].apply(
            map_values, args = (values_dict_M,))
    return df


def add_car_value(df):
    ## adding only from region of claim ##
	cars = pd.read_excel(
	    "./data/cars_brand/Statista_eu-car-sales-average-prices.xlsx"
	)  
	cars = cars[['ColonnaDB','Price_2017']]
	carmodels = cars[~cars.ColonnaDB.str.contains("vehicle_model")]
	fp_carmodels = carmodels[~carmodels.ColonnaDB.str.contains("tp")]
	tp_carmodels = carmodels[~carmodels.ColonnaDB.str.contains("fp")]
    #filling na with mean
	fp_carmodels = fp_carmodels.fillna(fp_carmodels.mean())    
	tp_carmodels = tp_carmodels.fillna(tp_carmodels.mean()) 

	tpcm = tp_carmodels.set_index('ColonnaDB')['Price_2017'].to_dict()
	fpcm = fp_carmodels.set_index('ColonnaDB')['Price_2017'].to_dict()
    #filling none with - 1
	tpcm['tp__vehicle_make__none'] = - 1
	fpcm['fp__vehicle_make__none'] = - 1
    
	df["tp_car_price"] = 0
	for i in range(len(df)):
		for item in tpcm:
			if df[item].iloc[i]==1:
				df.at[i, "tp_car_price"] = tpcm[item]
	df["fp_car_price"] = 0
	for i in range(len(df)):
		for item in fpcm:
			if df[item].iloc[i]==1:
				df.at[i, "fp_car_price"] = fpcm[item]
    
	return df


#### Nice To Have ###
##ragionare sulle singole regioni presenti e non su tutto

"""
Definizione ISTAT: https://www.tuttitalia.it/statistiche/nord-centro-mezzogiorno-italia/
NordOvest: liguria, lombardia, piemonte
NordEst: liguria, emiliaromagna
Centro: lazio, toscana
Mezzogiorno: campania, sicilia

region_of_claim__lombardia
region_of_claim__piemonte

region_of_claim__veneto
region_of_claim__emiliaromagna

region_of_claim__toscana
region_of_claim__lazio

region_of_claim__puglia
region_of_claim__campania
region_of_claim__sicilia

region_of_claim__other	


region_of_policy__liguria							
region_of_policy__lombardia
region_of_policy__piemonte

region_of_policy__veneto
region_of_policy__emiliaromagna	

region_of_policy__toscana
region_of_policy__lazio	

region_of_policy__campania	
region_of_policy__sicilia

region_of_policy__none
region_of_policy__other

"""
