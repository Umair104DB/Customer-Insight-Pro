from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Customer Inside Pro", page_icon="📊", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}.hero{padding:24px;border-radius:16px;background:linear-gradient(120deg,#10243a,#1b4965);color:white}div[data-testid='stMetric']{background:#f5f7fa;border:1px solid #e3e8ef;padding:12px;border-radius:12px}</style>", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>Customer Inside Pro</h1><p>Explore customer behaviour and identify customers who may leave or stay using a machine-learning model.</p></div>",unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent/"data"/"customer_churn_demo.csv")
df=load_data()
st.sidebar.title("Workspace")
page=st.sidebar.radio("Navigate",["Overview","Customer explorer","Risk scorer","Model notes"])
st.sidebar.caption("Portfolio demo using synthetic sample records.")
features=[c for c in df.columns if c not in ["customer_id","churn"]]
cat=[c for c in features if df[c].dtype=="object"]
num=[c for c in features if c not in cat]
X=df[features]; y=df.churn.map({"No":0,"Yes":1})
prep=ColumnTransformer([("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("category",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model.fit(Xtr,ytr); pred=model.predict(Xte); probs=model.predict_proba(Xte)[:,1]
scores={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,probs)}
if page=="Overview":
    a,b,c,d=st.columns(4)
    a.metric("Sample customers",len(df)); b.metric("Demo churn rate",f"{(df.churn=='Yes').mean():.1%}")
    c.metric("Avg monthly charge",f"${df.monthly_charges.mean():.2f}"); d.metric("Avg tenure",f"{df.tenure_months.mean():.1f} months")
    left,right=st.columns(2)
    with left:
        rates=df.groupby("contract")["churn"].apply(lambda s:(s=="Yes").mean()).reset_index(name="rate")
        st.plotly_chart(px.bar(rates,x="contract",y="rate",color="contract",title="Churn rate by contract",labels={"rate":"Churn rate"}).update_layout(showlegend=False,yaxis_tickformat=".0%"),use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(df,x="tenure_months",color="churn",barmode="overlay",title="Tenure distribution"),use_container_width=True)
    st.subheader("Held-out model metrics")
    cols=st.columns(5)
    for col,(name,val) in zip(cols,scores.items()): col.metric(name,f"{val:.3f}")
    st.caption("Metrics are illustrative and based on a small synthetic dataset, not real customer records.")
elif page=="Customer explorer":
    st.subheader("Filter and explore customer records")
    a,b,c=st.columns(3)
    contracts=a.multiselect("Contract",sorted(df.contract.unique()),default=sorted(df.contract.unique()))
    labels=b.multiselect("Churn label",["Yes","No"],default=["Yes","No"])
    span=c.slider("Tenure months",0,72,(0,72))
    view=df[df.contract.isin(contracts)&df.churn.isin(labels)&df.tenure_months.between(*span)]
    st.write(f"{len(view)} records match the filters.")
    st.dataframe(view,hide_index=True,use_container_width=True)
    st.download_button("Download filtered CSV",view.to_csv(index=False).encode(),"filtered_customers.csv","text/csv")
    st.plotly_chart(px.scatter(view,x="tenure_months",y="monthly_charges",color="churn",hover_data=["contract","internet_service"],title="Tenure and monthly charges"),use_container_width=True)
elif page=="Risk scorer":
    st.subheader("Customer Status Prediction")
    with st.form("score"):
        a,b,c=st.columns(3)
        tenure=a.slider("Tenure (months)",0,72,12)
        monthly=b.slider("Monthly charges ($)",20.0,130.0,75.0)
        contract=c.selectbox("Contract",sorted(df.contract.unique()))
        internet=a.selectbox("Internet service",sorted(df.internet_service.unique()))
        support=b.selectbox("Tech support",sorted(df.tech_support.unique()))
        payment=c.selectbox("Payment method",sorted(df.payment_method.unique()))
        paperless=a.selectbox("Paperless billing",sorted(df.paperless_billing.unique()))
        go=st.form_submit_button("Predict Customer Status",type="primary")
    if go:
        sample=pd.DataFrame([{"tenure_months":tenure,"monthly_charges":monthly,"contract":contract,"internet_service":internet,"tech_support":support,"payment_method":payment,"paperless_billing":paperless}])
        risk=float(model.predict_proba(sample)[0,1])
        st.metric("Customer Status", "Customer Left Risk" if risk >= .5 else "Customer Stays", f"Estimated probability: {risk:.1%}")
        st.progress(risk)
        st.info("Use this score as a demonstration only. The model was trained on synthetic sample data.")
else:
    st.subheader("How it works")
    st.markdown("""
    - Cleans missing values with median/mode imputers.
    - Scales numeric fields and one-hot encodes categories.
    - Trains a Logistic Regression classifier on a stratified train/test split.
    - Reports accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.

    Production checklist: use a licensed representative dataset, validate target definitions, compare cross-validated models, calibrate probabilities, monitor drift and assess fairness.
    """)
    st.plotly_chart(px.imshow(confusion_matrix(yte,pred,labels=[0,1]),text_auto=True,x=["Customer Stays","Customer Left Risk"],y=["Actual: Customer Stays","Actual: Customer Left Risk"],title="Held-out confusion matrix",color_continuous_scale="Blues"),use_container_width=True)
