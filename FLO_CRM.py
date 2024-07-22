import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
pd.set_option('display.width', 500)


df = pd.read_csv(r"C:\Users\huawei\Desktop\miuul\FLOMusteriSegmentasyonu\flo_data_20k.csv")
df.head(5)
df.shape
df.info()
df.isnull().sum()
df.describe().T

df["first_order_date"] = pd.to_datetime(df['first_order_date'])
df["last_order_date"] = pd.to_datetime(df['last_order_date'])
df["last_order_date_online"] = pd.to_datetime(df['last_order_date_online'])
df["last_order_date_offline"] = pd.to_datetime(df['last_order_date_offline'])

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.groupby("order_channel").agg({"master_id":"count",
                                 "total_order":[sum, "mean"],
                                 "total_value":[sum, "mean"]})


df.sort_values(by = "total_value", ascending=False).head(5)
df.sort_values(by = "total_order", ascending=False).head(5)


rfm = pd.DataFrame()

analysis_date = pd.to_datetime("2021-6-1")

rfm["master_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).days
rfm["frequeny"] = df["total_order"]
rfm["monetary"] = df["total_value"]
rfm.head()
rfm.describe().T

rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["f_score"] = pd.qcut(rfm["frequeny"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])
rfm["rfm"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
rfm["rf_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str)

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequeny", "monetary"]].groupby("segment").agg(["mean", "count"])

dff = pd.merge(df, rfm, on="master_id", how="inner")
dff = dff.drop("customer_id", axis=1)

target_1 = dff[dff["segment"].isin(["cant_loose","hibernating","new_customers"])
            & dff["interested_in_categories_12"].str.contains("ERKEK")|dff["interested_in_categories_12"].str.contains("COCUK")]

target_1.head(5)

target_2 = dff[dff["segment"].isin(["champions","loyal_customers"]) & dff["interested_in_categories_12"].str.contains("KADIN")]
target_2.head(5)


###################################
###################################
###################################
###################################

import pandas as pd
import numpy as np
pip install lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
pd.set_option('display.width', 500)


df = pd.read_csv(r"C:\Users\huawei\Desktop\miuul\FLOMusteriSegmentasyonu\flo_data_20k.csv")
df.head(5)
df.shape
df.info()
df.isnull().sum()
df.describe().T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

outlier_thresholds(df)

num_col = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in num_col:
    print(outlier_thresholds(df, col))

df.describe().T

for col in num_col:
    replace_with_thresholds(df, col)

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["first_order_date"] = pd.to_datetime(df['first_order_date'])
df["last_order_date"] = pd.to_datetime(df['last_order_date'])
df["last_order_date_online"] = pd.to_datetime(df['last_order_date_online'])
df["last_order_date_offline"] = pd.to_datetime(df['last_order_date_offline'])

analysis_date = pd.to_datetime("2021-6-1")


cltv = pd.DataFrame()
cltv["master_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
cltv["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.days) / 7
cltv["frequency"] = df["total_order"]
cltv["monetary_cltv_avg"] = df["total_value"] / df["total_order"]

cltv.head(5)
cltv.describe().T

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

cltv["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv['frequency'],
                                       cltv['recency_cltv_weekly'],
                                       cltv['T_weekly'])

cltv.sort_values("exp_sales_3_month",ascending=False)[:10]

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])
cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                cltv['monetary_cltv_avg'])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency_cltv_weekly'],
                                   cltv['T_weekly'],
                                   cltv['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv["cltv"] = cltv

cltv.head()
# cltv = 395.73 --> bu müşterinin verilen sürede bana getireceği tahmin kazanç


# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv.sort_values("cltv",ascending=False)[:20]

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerinizi
# 4 gruba (segmente) ayırınız
# ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 2. CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
# Daha az mı ya da daha çok mu olmalıdır. Yorumlayınız.


# 3. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa
# 6 aylık aksiyon önerilerinde bulununuz


# segment bazında istatistikleri görmek istersek
cltv_df.groupby("cltv_segment").describe().T

###############################################################
# BONUS: Tüm süreci fonksiyonlaştırınız.
###############################################################