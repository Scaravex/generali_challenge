import os 
import pandas as pd
from src.utils import fetch_data, reduce_mem_usage
from src.utils import drop_columns_without_variability, drop_artificial_columns
from src.utils import calcola_complessita_sinistro, calculate_diff_age
from src.utils import getDummies, correlation
from src.regions_cars_extractor import adding_macro_region, add_istat_kpis, add_car_value
from src.utils import lasso_ftrs_w, get_covariate_weights

current_dir = os.getcwd()
main_path = os.path.dirname(current_dir)
# main_path =  r'C:\Users\Marco\Documents\GitHub\axa_challenge'
os.chdir(main_path)

# Importing data
training = fetch_data("train")
test = fetch_data("validation")

#reducing memory
training = reduce_mem_usage(training)
test = reduce_mem_usage(test)

# Applica tutte le funzioni di feature engineering definite in utils_feature_engineering


# aggiunge due colonne: Macro_region_of_policy e Macro_region_of_claim

one_hot_macro_region = 0
test = adding_macro_region(test, one_hot_macro_region, one_hot_macro_region) # flag one eliminates info for regions
training = adding_macro_region(training, one_hot_macro_region, one_hot_macro_region) # flag one eliminates info for regions

# adding istat data: mortalità, pericolositò, gravità by region
training = add_istat_kpis(training)
test     = add_istat_kpis(test)

# creating dummies out
if one_hot_macro_region:
    one_hot,cols = getDummies(training,"Macro_region_of_claim")
    training = training.join(one_hot)
    training = training.drop('Macro_region_of_claim',axis = 1)
    one_hot,cols = getDummies(test,"Macro_region_of_claim")
    test = test.join(one_hot)
    test = test.drop('Macro_region_of_claim',axis = 1)
    
    one_hot, cols = getDummies(training,"Macro_region_of_policy")
    training = training.join(one_hot)
    training = training.drop('Macro_region_of_policy',axis = 1)
    one_hot, cols = getDummies(test,"Macro_region_of_policy")
    test['Macro_region_of_policy_OTHER'] = 0
    test = test.join(one_hot)
    test = test.drop('Macro_region_of_policy',axis = 1)
else:
    training = training.drop('Macro_region_of_claim',axis = 1)
    test = test.drop('Macro_region_of_claim',axis = 1)
    #training = training.drop('Macro_region_of_policy',axis = 1)
    #test = test.drop('Macro_region_of_policy',axis = 1)
    
# pricing cars --> livello macchine 
training = add_car_value(training)
test = add_car_value(test)


###Eliminate row with boat
training = training.drop(training.loc[training.tp__vehicle_type__boat==True].index,axis=0)
training = training.drop("tp__vehicle_type__boat",axis=1)
test = test.drop("tp__vehicle_type__boat",axis=1)

# eliminate is_thief_known
training = training.drop("is_thief_known",axis=1)
test = test.drop("is_thief_known",axis=1)


# aggiungo features di similarità al test set
training["weights_test"] = get_covariate_weights(training,lasso_ftrs_w,False)
test["weights_test"]= get_covariate_weights(test,lasso_ftrs_w,False)


# drop empty columns
training = drop_columns_without_variability(training)
test = drop_columns_without_variability(test)

# if distance between the two is near

# KPI che da un punteggio alla complessità del sinistro
training = calcola_complessita_sinistro(training)
test = calcola_complessita_sinistro(test)

# differenza di etò
training = calculate_diff_age(training)
test = calculate_diff_age(test)


#vive nella stessa città




# togliere dummy specifiche cars(non generalizzo)
# you should add other = 1 --> otherwise rows not consistent
training['tp__vehicle_model__other'] = training.apply(lambda x: True if 
        x["tp__vehicle_model__agila_2__serie_opel_agila_12_16v_86cv_enjoy"]==1 else False, axis=1)
training['tp__vehicle_model__other'] = training.apply(lambda x: True if 
        x["tp__vehicle_model__yaris_3__serie_toyota_yaris_10_5_porte_active"]==1 else False, axis=1)

test['tp__vehicle_model__other'] = test.apply(lambda x: True if 
        x["tp__vehicle_model__agila_2__serie_opel_agila_12_16v_86cv_enjoy"]==1 else False, axis=1)
test['tp__vehicle_model__other'] = test.apply(lambda x: True if 
        x["tp__vehicle_model__yaris_3__serie_toyota_yaris_10_5_porte_active"]==1 else False, axis=1)

training.drop(["tp__vehicle_model__agila_2__serie_opel_agila_12_16v_86cv_enjoy",
               "tp__vehicle_model__yaris_3__serie_toyota_yaris_10_5_porte_active"],
                axis=1,inplace=True) # 2 in training e 2 in test
test.drop(["tp__vehicle_model__agila_2__serie_opel_agila_12_16v_86cv_enjoy",
               "tp__vehicle_model__yaris_3__serie_toyota_yaris_10_5_porte_active"],
                axis=1,inplace=True) # 2 in training e 2 in test

# drop columns with high correlation with another one
# removing highly correlated features >0.999
corr_cols_test = correlation(test, 0.99)

training = training.drop(corr_cols_test,axis = 1)
test = test.drop(corr_cols_test,axis = 1)

#eliminate variables with less than 1% different feature

# drop artificial columns --> reduce accuracy. skip for now
# training = drop_artificial_columns(training)
# test     = drop_artificial_columns(test)


# Scrive a CSV i file con feature engineering. 

training.to_csv("./data/training_fe.csv", index = False)
test.to_csv("./data/validation_fe.csv", index = False)
# Idee per feature selection:
# lasso regression
# provare a creare una colonna normale 0,1 e vedere tutte le variabili con feature iumportance inferiore
# Analisi di correlazione

# Idee per feature engineering
# Totale di flag=True per ciascuna osservazione

