import streamlit as st

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler

import pickle

from sklearn import metrics

st.title("Sistem Klasifikasi Data")
st.write("""
# Web Apps - Klasifikasi Segmentasi pelanggan
Applikasi Berbasis Web untuk Mengklasifikasi **Segmentasi pelanggan**
""")

tab1, tab2, tab3 = st.tabs(["Data Understanding", "Preprocecing", "Implementation"])

with tab1:
    st.markdown(""" <a href=
    https://github.com/arsymaulanaali/hosting-dataset-heroku>Link Github </a>
    """,unsafe_allow_html=True)
    st.markdown(""" <a href=
    https://www.kaggle.com/datasets/vetrirah/customer>Link Data </a>
    """,unsafe_allow_html=True)

    st.subheader("Dataset")

    df = pd.read_csv("Train.csv")
    st.write(df)
    st.markdown(""" <ol>
        <li> kolom ID = merupakan isi dari nomer id pelanggan</li>
        <li> kolom Gender = merupakan isi dari jenis kelamin pelanggan</li>
        <li> kolom Ever_Married = merupakan isi dari status menikah pelanggan</li>
        <li> kolom Age = merupakan isi dari umur pelanggan</li>
        <li> kolom Graduate = merupakan isi dari  status pendidikan pelanggan</li>
        <li> kolom Profession = merupakan isi dari pekerjaan pelanggan</li>
        <li> kolom Work_Experience = merupakan isi dari pengalaman kerja pelanggan selama  per tahun</li>
        <li> kolom Spending_Score = merupakan isi dari skor pengeluaran pelanggan</li>
        <li> kolom Family_Size = merupakan isi dari jumlah anggota keluarga pelanggan(termasuk pelanggan)</li>
        <li> kolom Var_1 = merupakan isi dari kategori anonim untk pelanggan</li>
        <li> kolom Segmentation = merupakan isi dari jenis klasifikasi pelanggan</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h5>Preprocessing Data</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <p style="text-align: justify;text-indent: 45px;">Preprocessing data adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur. Preprocessing melibatkan proses validasi dan imputasi data.</p>
    <p style="text-align: justify;text-indent: 45px;">Salah satu tahap Preprocessing data adalah Normalisasi. Normalisasi data adalah elemen dasar data mining untuk memastikan record pada dataset tetap konsisten. Dalam proses normalisasi diperlukan transformasi data atau mengubah data asli menjadi format yang memungkinkan pemrosesan data yang efisien.</p>
    <br>
    """,unsafe_allow_html=True)
    for i in range(len(df["ID"])):
        if df["Gender"][i]=='Male':
            df["Gender"][i]=0
        elif df["Gender"][i]=='Female':
            df["Gender"][i]=1

        if df["Ever_Married"][i]=='No':
            df["Ever_Married"][i]=0
        elif df["Ever_Married"][i]=='Yes':
            df["Ever_Married"][i]=1

        if df["Graduated"][i]=='No':
            df["Graduated"][i]=0
        elif df["Graduated"][i]=='Yes':
            df["Graduated"][i]=1

        if df["Profession"][i]=='Artist':
            df["Profession"][i]=0
        elif df["Profession"][i]=='Healthcare':
            df["Profession"][i]=1
        elif df["Profession"][i]=='Entertainment':
            df["Profession"][i]=2
        elif df["Profession"][i]=='Engineer':
            df["Profession"][i]=3
        elif df["Profession"][i]=='Doctor':
            df["Profession"][i]=4
        elif df["Profession"][i]=='Lawyer':
            df["Profession"][i]=5
        elif df["Profession"][i]=='Executive':
            df["Profession"][i]=6
        elif df["Profession"][i]=='Marketing':
            df["Profession"][i]=7
        elif df["Profession"][i]=='Homemaker':
            df["Profession"][i]=8

        if df["Spending_Score"][i]=='Low':
            df["Spending_Score"][i]=0
        elif df["Spending_Score"][i]=='Average':
            df["Spending_Score"][i]=1
        elif df["Spending_Score"][i]=='High':
            df["Spending_Score"][i]=2

        if df["Var_1"][i]=='Cat_1':
            df["Var_1"][i]=1
        elif df["Var_1"][i]=='Cat_2':
            df["Var_1"][i]=2
        elif df["Var_1"][i]=='Cat_3':
            df["Var_1"][i]=3
        elif df["Var_1"][i]=='Cat_4':
            df["Var_1"][i]=4
        elif df["Var_1"][i]=='Cat_5':
            df["Var_1"][i]=5
        elif df["Var_1"][i]=='Cat_6':
            df["Var_1"][i]=6
        elif df["Var_1"][i]=='Cat_7':
            df["Var_1"][i]=7
    
    scaler = st.radio(
    "Pilih metode normalisasi data",
    ('Tanpa Scaler', 'MinMax Scaler'))

    df_drop_id=df.drop(['ID'], axis=1)

    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df_drop_id
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df_drop_id, columns = ['Age'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['age'])
        df_drop_column_for_minmaxscaler=df_drop_id.drop(['Age'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    df_new.dropna(inplace=True,axis=0)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    st.subheader("Parameter Inputan")

    sex=st.selectbox(
        'Pilih Jenis Kelamin',
        ['Laki-laki','Perempuan']
    )
    if sex=='Laki-laki':
        sex=0
    elif sex=='Perempuan':
        sex=1

    married=st.selectbox(
        'Status Pernikahan',
        ['Sudah Menikah','Belum Menikah']
    )
    if married=='Sudah Menikah':
        married=1
    elif married=='Belum Menikah':
        married=0

    umur=st.number_input('Umur',1)

    graduated=st.selectbox(
        'Pernah Bersekolah',
        ['Tamat','Belum Tamat']
    )
    if graduated=='Belum Tamat':
        graduated=0
    elif graduated=='Tamat':
        graduated=1

    pekerjaan=st.selectbox(
        'Pekerjaan',
        ['Artis','Tenaga Kesehatan','Entertaiment','Engineer','Dokter','Pengacara','Executive','Pedagang','Kuli']
    )
    if pekerjaan=='Artis':
        pekerjaan=0
    elif pekerjaan=='Tenaga Kesehatan':
        pekerjaan=1
    elif pekerjaan=='Entertaiment':
        pekerjaan=2
    elif pekerjaan=='Engineer':
        pekerjaan=3
    elif pekerjaan=='Dokter':
        pekerjaan=4
    elif pekerjaan=='Pengacara':
        pekerjaan=5
    elif pekerjaan=='Executive':
        pekerjaan=6
    elif pekerjaan=='Pedagang':
        pekerjaan=7
    elif pekerjaan=='Kuli':
        pekerjaan=8

    pengalaman_kerja=st.number_input('Lama Berdasarkan Berapa Tahun')

    spending_score=st.selectbox(
        'Rating Dari Pelanggan',
        ['Rendah','Sedang','Tinggi']
    )
    if spending_score=='Rendah':
        spending_score=0
    elif spending_score=='Sedang':
        spending_score=1
    elif spending_score=='Tinggi':
        spending_score=2
    family_size=st.number_input('jumlah anggota keluarga')
    var_1=st.selectbox(
        'Kategori Anonim untuk pelanggan',
        ['cat_1','cat_2','cat_3','cat_4','cat_5','cat_6','cat_7']
    )
    if var_1=='cat_1':
        var_1=1
    elif var_1=='cat_2':
        var_1=2
    elif var_1=='cat_3':
        var_1=3
    elif var_1=='cat_4':
        var_1=4
    elif var_1=='cat_5':
        var_1=5
    elif var_1=='cat_6':
        var_1=6
    elif var_1=='cat_7':
        var_1=7

    algoritma = st.selectbox(
        'pilih algoritma klasifikasi',
        ('KNN','Naive Bayes','Random Forest')
    )
    submit=st.button('submit')
    X=df_new.iloc[:,0:9].values
    y=df_new.iloc[:,9].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,stratify=y, random_state=42)
    if submit:
        if algoritma=='KNN':
            model=KNeighborsClassifier(n_neighbors=3)
            filename='knn.pkl'
        elif algoritma=='Naive Bayes':
            model = GaussianNB()
            filename='gaussian.pkl'
        elif algoritma=='Random Forest':
            model = RandomForestClassifier(n_estimators = 100) 
            filename='randomforest.pkl'

        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 

        from sklearn import metrics
        score=metrics.accuracy_score(y_test,Y_pred)

        import pickle
            
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))

        if scaler == 'Tanpa Scaler':
            dataArray = [sex, married, umur, graduated, pekerjaan, pengalaman_kerja, spending_score,family_size, var_1]
        else:
            umur_proceced = (umur - df['Age'].min(axis=0)) / (df['Age'].max(axis=0) - df['Age'].min(axis=0))
            dataArray = [umur_proceced, sex, married, graduated, pekerjaan, pengalaman_kerja, spending_score,family_size, var_1]
            
        pred = loaded_model.predict([dataArray])

        # apply the whole pipeline to data
        pred = loaded_model.predict([dataArray])

        st.success(f"hasil prediksi : {pred[0]}")
        st.success(f"hasil akurasi : {score}")