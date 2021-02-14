import pandas as pd
import numpy as np
import string
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt 
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/ffmpeg/bin/ffmpeg' 

from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta, date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize

Punct_List = dict((ord(punct), None) for punct in string.punctuation + '¿¡')

def TxNormalize(text):
    
    return word_tokenize(text.lower().translate(Punct_List))

def coincidir(texto):
    Tokens_List.append(texto)
    TfidfVec = TfidfVectorizer(tokenizer = TxNormalize) 
    tfidf = TfidfVec.fit_transform(Tokens_List)
    Tokens_List.remove(texto)
    vals = cosine_similarity(tfidf[-1], tfidf)
    flat = vals.flatten()
    flat.sort()
    
    return Tokens_List[vals.argsort()[0][-2]]

def nice_axes(ax, ax2, vals, a, b):
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('grey')
    ax.set_xticks(vals)
    ax.set_xticklabels(['{:3.1f} %'.format(x) if (x < a and x != 0)  else '{:3.1f} %'.format(vals.max() - x) if (vals.max() - x < - b and  vals.max() - x != 0) else None  for x in vals]
                       , fontsize = 18)
    
    ax.grid(axis='x', alpha=0.25)
    ax.yaxis.set_tick_params(labelsize=18)
    
    ax2.set_xticks(vals*-1)
    [spine.set_visible(False) for spine in ax2.spines.values()]
    ax2.set_xticklabels('')
    ax2.tick_params(left = False, top = False, bottom = False)
    ax.tick_params(left = False, bottom = False)
    
def pivot_table(df, val):

    return (df
            .groupby('Comunidad')
            .resample('w')
            .agg({val:'max'})
            .reset_index()
            .pivot(index = 'Fecha', columns='Comunidad', values= val)
            .fillna(method='ffill')
            .fillna(0)
            .reset_index()
            )

def prepare_data(df, val, steps = 5):
    
    df = pivot_table(df, val)
    df.index = df.index * steps
    last_idx = df.index[-1] + 1
    
    df = df.reindex(range(last_idx))
    df.Fecha = df.Fecha.fillna(method='ffill')
    df = df.set_index('Fecha')
    df_rank = df.rank(axis=1, method='first')
    df = df.interpolate()
    df_rank = df_rank.interpolate()
    
    return df, df_rank

url = 'https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/documentos/Informe_Comunicacion_'
df = None
for i in range(100):
    try:
        df = (pd.read_excel(url + (datetime.now() - timedelta(days = + i)).strftime("%Y%m%d") + '.ods', engine="odf")
              .rename(columns = {'Dosis entregadas Pfizer (1)':'Dosis Pfizer'
                                 , 'Dosis entregadas (1)':'Dosis Pfizer'
                                 , 'Dosis entregadas AstraZeneca (1)': 'Dosis AstraZeneca'
                                 , 'Dosis entregadas Moderna (1)': 'Dosis Moderna'
                                 , 'Total Dosis entregadas (1)': 'Dosis'
                                 , 'Dosis administradas (2)': 'Puestas'
                                 , '% sobre entregadas': 'Usadas'
                                 , 'Unnamed: 0': 'Comunidad'
                                 , 'Nº Personas vacunadas(pauta completada)':'Pauta Completa'
                                 , 'Fecha de la última vacuna registrada (2)':'Fecha'
                            })
              .set_index('Fecha')
              .loc[lambda df: df.Comunidad != 'Totales']
              .append(df)
              .fillna(0))
        
    except:
        pass

url = 'https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/2853.csv?nocab=1'

poblacion = pd.read_csv(url, error_bad_lines=False, sep=';', header=0)

poblacion = poblacion[(poblacion['Comunidades y Ciudades Autónomas'] != 'Total') 
                      & (poblacion['Periodo'] == 2020)].pivot(index = 'Comunidades y Ciudades Autónomas'
                                                             , columns = 'Sexo'
                                                             , values = 'Total')

Tokens_List = sent_tokenize(" ".join(x + '.' for x in df.Comunidad.unique()), 'spanish')
matriz = {}

for i in poblacion.index.values:
    matriz[i] = coincidir(i.replace('Balears','Baleares'))[:-1]
            
poblacion.index = poblacion.index.map(matriz)

df = (df.reset_index()
       .set_index('Comunidad')
       .merge(poblacion, left_index=True, right_index=True)
       .reset_index()
       .set_index('Fecha')
       .drop(['Hombres','Mujeres'], axis = 1)
       .rename(columns = {'index': 'Comunidad'})
      )

df = (df
       .assign(Vacunados = np.array((df.Puestas - df['Pauta Completa']) / df.Total.str.replace('.','').astype(int) * 100))
       .assign(Inmunizados = np.array(df['Pauta Completa'] / df.Total.str.replace('.','').astype(int) * 100))
       .assign(Entregadas = np.array((-df['Dosis Pfizer'] - df['Dosis Moderna'] - df['Dosis AstraZeneca'] + df.Puestas)/ df.Total.str.replace('.','').astype(int) * 100))
       .loc[ :date.today().isoformat()]
      )

Datos, rk = prepare_data(df, 'Vacunados', steps = 5)

colors = plt.cm.tab20(range(len(df.Comunidad.unique())))
date = df.index.sort_values(ascending = True).max().isoformat()

vals = np.linspace(start = 0
                   , stop = int(df.loc[date].Vacunados.max() - df.loc[date].Entregadas.min()) + 1
                   , num = int(df.loc[date].Vacunados.max() - df.loc[date].Entregadas.min()) * 2 +3)

def init():
    ax.clear()
    nice_axes(ax, ax2, vals, df.loc[date].Vacunados.max(), df.loc[date].Entregadas.min())

def update(i):
    for bar in ax.containers:
        bar.remove()

    ax.barh(y = rk.iloc[i]
            , width = Datos.iloc[i].values
            , color = colors
            , tick_label = Datos.columns
            )

fig = plt.Figure(figsize = (20,10), dpi = 320)
ax = fig.add_subplot()
ax2 = ax.twiny()

anim = FuncAnimation(fig = fig
                    , func = update
                    , init_func = init
                    , frames = len(Datos)
                    , interval = 300
                    , repeat = False)

anim.save('covid19.mp4')