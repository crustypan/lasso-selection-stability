"""Fetch the original UCI workbook; derive a reproducible development/test split."""
import hashlib
import io
import zipfile
from datetime import datetime, timezone
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CFG, ROOT, OUT, atomic_json

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW = ROOT/"data/raw"
STATUS = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1,7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1,7)]
NAMES = STATUS+BILLS+PAYMENTS
MONTHS = ["Sep","Aug","Jul","Jun","May","Apr"]
TARGET = "default payment next month"


def workbook():
    RAW.mkdir(parents=True,exist_ok=True)
    archive = RAW/"uci350.zip"
    if not archive.exists():
        with urlopen(URL,timeout=90) as response:
            blob = response.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as check:
            assert check.testzip() is None
        archive.write_bytes(blob)
        atomic_json(RAW/"download.json",dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),url=URL))
    with zipfile.ZipFile(archive) as z:
        member, = [f for f in z.namelist() if f.lower().endswith(".xls")]
        body = z.read(member)
    df = pd.read_excel(io.BytesIO(body),header=1,engine="xlrd")
    assert len(df)==30000 and df.ID.is_unique and not df.ID.isna().any()
    assert set(df[TARGET])=={0,1} and not df[NAMES].isna().any().any()
    hashes = dict(zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  xls_sha256=hashlib.sha256(body).hexdigest(), member=member)
    return df,hashes


def split(df):
    y = df[TARGET].to_numpy()
    dev,test = train_test_split(np.arange(len(df)),test_size=CFG["credit_test_fraction"],
                                stratify=y,random_state=CFG["credit_split_seed"])
    pilot,validation = train_test_split(dev,test_size=CFG["credit_validation_fraction"],
                               stratify=y[dev],random_state=CFG["credit_validation_seed"])
    assert set(dev).isdisjoint(test) and set(pilot).isdisjoint(validation)
    return dev,test,pilot,validation


def design(df, raw_status=False):
    x = df[NAMES].to_numpy(dtype=float,copy=True)
    if not raw_status:
        x[:,:6] = np.maximum(0,x[:,:6])
    return x


def prepare():
    import json
    df,h = workbook()
    dev,test,pilot,val = split(df)
    y=df[TARGET].to_numpy()
    splitfile = ROOT/"data/split_ids.npz"
    np.savez_compressed(splitfile,**{k:df.ID.to_numpy()[v] for k,v in
                                     dict(development=dev,test=test,pilot=pilot,validation=val).items()})
    mapping = [dict(uci=f"X{6+i}",field=name,month=MONTHS[i%6]+" 2005",
                    family=["status","bill","repayment"][i//6]) for i,name in enumerate(NAMES)]
    status = {k:{str(a):int(b) for a,b in df[k].value_counts().sort_index().items()} for k in STATUS}
    atomic_json(OUT/"credit_data_audit.json",dict(**h,**json.loads((RAW/"download.json").read_text()),
        rows=len(df),unique_ids=int(df.ID.nunique()),target=TARGET,target_counts=df[TARGET].value_counts().to_dict(),
        split_sizes=[len(a) for a in (dev,test,pilot,val)],mapping=mapping,status_counts=status,
        fields=list(df.columns),missing_model_entries=int(df[NAMES].isna().sum().sum()),license="CC BY 4.0",
        doi="10.24432/C55S3H"))
    x=pd.DataFrame(design(df)[dev],columns=NAMES)
    pairs=[]
    for group,names in zip(["status","bill","repayment"],[STATUS,BILLS,PAYMENTS]):
        cor=x[names].corr(method="spearman")
        for i,a in enumerate(names):
            for b in names[i+1:]:
                pairs.append(dict(group=group,first=a,second=b,spearman=cor.loc[a,b]))
    pd.DataFrame(pairs).to_csv(OUT/"credit_development_correlations.csv",index=False)
    print("Verified",len(df),"customer IDs. Development/test sizes:",len(dev),len(test))


if __name__ == "__main__":
    prepare()
