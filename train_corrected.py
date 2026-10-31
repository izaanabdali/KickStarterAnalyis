"""Leakage-safe metadata baseline. Historical notebooks are retained for comparison."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',default='ks-projects-201801.csv');parser.add_argument('--output',default='corrected_results.json');args=parser.parse_args()
 df=pd.read_csv(args.data)
 # Restrict to terminal success/failure outcomes; cancelled/live projects excluded.
 df=df[df.state.isin(['successful','failed'])].copy()
 df['campaign_length']=(pd.to_datetime(df.deadline)-pd.to_datetime(df.launched)).dt.total_seconds()/86400
 df['launch_month']=pd.to_datetime(df.launched).dt.month
 # No pledged/backer amounts or global target-encoded success-rate features.
 numeric=['usd_goal_real','campaign_length','launch_month'];categorical=['main_category','country']
 X=df[numeric+categorical];y=df.state.eq('successful').astype(int)
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
 results={}
 for name,model in [('logistic_regression',LogisticRegression(max_iter=2000,random_state=42)),('random_forest',RandomForestClassifier(n_estimators=100,max_depth=12,n_jobs=2,random_state=42)),('gradient_boosting',GradientBoostingClassifier(random_state=42))]:
  prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
  pipe=Pipeline([('prepare',prep),('model',model)]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
  results[name]={'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0)}
  print(name,results[name],flush=True)
 result={'evaluation':'stratified 80/20 holdout; seed 42; successful class positive','rows':len(df),'majority_baseline_accuracy':float(max(yte.mean(),1-yte.mean())),'results':results,'limitations':'Single random holdout, not temporal validation. Original historical metrics are not comparable because outcome filtering and features differ.'}
 Path(args.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
