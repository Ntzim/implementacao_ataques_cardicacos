import pandas as pd
import streamlit as st
import numpy as np
import joblib




# Técnicas
from sklearn.neighbors import KNeighborsClassifier as KNC




#Home Page
st.sidebar.header('Escolha uma pagina')

escolhas = ['Pagina Inicial','Previsão de Ataques Cardiacos']
escolha_do_indicadores = st.sidebar.selectbox('Selecione a página que você quer ver :',escolhas)


# DEFININDO FUNÇÃO
def get_user_data():

    #
    st.sidebar.header('Preencha os campos com os resultados do exame')


    #Input Nome Paciente
    user_input = st.sidebar.text_input("Nome do paciente")

    # Input Idade
    age = st.sidebar.number_input("Idade", 18, 100, 25)

    #Input Sexo
    sex_options = {"Masculino": "M", "Feminino": "F"}
    sex = st.sidebar.radio("Sexo", list(sex_options.keys()))

    #Input Tipo de Dor no Peito
    chest_pain_type_options = {"Angina típica": "TA", "Angina típica": "ATA",
                               "Dor não anginosa": "NAP", "Assintomática": "ASY"}
    chest_pain_type = st.sidebar.radio("Tipo de Dor no Peito", list(chest_pain_type_options.keys()))

    #Input Pressão Arterial em Repouso
    resting_blood_pressure = st.sidebar.number_input("Pressão Arterial em Repouso", 0, 350, 120)

    #Input Colesterol
    cholesterol = st.sidebar.number_input('Colesterol', 0, 700, 200)

    #Input Glicemia em Jejum
    fasting_blood_sugar_options = {"Sim": 1, "Não": 0}
    fasting_blood_sugar = st.sidebar.radio("Glicemia em Jejum >120 mg ?", list(fasting_blood_sugar_options.keys()))

    #Input Eletrcardiograma
    resting_ecg_options = {"Normal": "Normal", "Com anormalidade das ondas(ST)": "ST",
                           "Mostrando Hipertrofia Ventricular(LVH)": "LVH"}
    resting_ecg = st.sidebar.radio("Resultado Eletrocardiograma em Repouso", list(resting_ecg_options.keys()))

    #Input Frequencia Max Cardiaca
    max_heart_rate_achieved = st.sidebar.number_input('Qual foi a frequência cardíaca maxima alcançada ?', 0, 300, 150)

    # Input Exercicio Induz Angina
    exercise_induced_angina_options = {"Sim": 1, "Não": 0}
    exercise_induced_angina = st.sidebar.radio("O exercicio induz a angina ?", list(exercise_induced_angina_options.keys()))

    #Input ST Depression
    st_depression = st.sidebar.number_input('Depressao ECG', -50, 100, 0)

    #Input ST SLpe
    st_slope_options = {"Up": "Up", "Flat": "Flat", "Down": "Down"}
    st_slope = st.sidebar.radio("Inclinação do Eletrocardiograma", list(st_slope_options.keys()))

    # Criando dicionario de dados
    user_data = {
        'Name':user_input,
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_blood_pressure,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_blood_sugar,
        'RestingECG': resting_ecg,
        'MaxHR': max_heart_rate_achieved,
        'ExerciseAngina': exercise_induced_angina,
        'Oldpeak': st_depression,
        'ST_Slope': st_slope
    }

    # Convertendo Dicionario em Dataframe
    user_data_df = pd.DataFrame(user_data, index=[0])
    
    return user_data_df



#PRÉ-PROCESSAMENTO DOS DADOS##



#IMPORTANDO MODELO

modelo = joblib.load('modelo/modelo_treinado_knn.pk')



# CONDICIONAIS

if escolha_do_indicadores == 'Pagina Inicial':
    st.markdown("<h1 style='color: red;'>Einsten Albert</h1>", unsafe_allow_html=True)
    st.subheader('Bem-Vindo ao sistema da Einstein Albert Labs')
    st.write('''O Grupo laboratórial e hospitalar EinstenAlbert é uma instituição líder em saúde cardiovascular, 
    nosso sistema de previsão de ataques cardíacos representa a ponte entre ciência, medicina e tecnologia. 
    Utilizamos técnicas de aprendizado de máquina para prever ataques cardíacos e instruímos os pacientes a tomar medidas preventivas para ter uma vida mais saudável.''')
    st.image('imagens/EinsteinAlbert.png')


elif escolha_do_indicadores == 'Previsão de Ataques Cardiacos':

    #SETANDO LOGO FIXA NO CANTO SUPERIOR DIREITO#

    st.markdown(
    """
    <style>
    .logo {
        position: fixed;
        top: 70px;
        right: 10px;
        width: 100px; 
        height: auto; 
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    #BUSCANDO IMAGEM
    st.markdown(
    '<img class="logo" src="https://static.vecteezy.com/ti/vetor-gratis/p3/509347-logotipo-para-uma-ilustracao-em-clinica-cardio-vetor.jpg" alt="Logo">',
    unsafe_allow_html=True
    )



    #TITULO
    st.subheader('Sistema de Previsão de Ataques Cardiacos')
    st.write('Nossa plataforma analisa cuidadosamente os dados do exame para prever possíveis ataques cardíacos. Com base nos resultados, fornecemos uma avaliação precisa do risco de ataque cardiaco. Sua saúde é nossa prioridade, e estamos aqui para oferecer orientações para a prevenção e o cuidado.')

    #DESCRIÇÃO
    st.write('Após fornecer todas as informações necessárias, clique no botão abaixo para gerar seu resultado personalizado.')

    #CHAMANDO FUNÇÃO
    df = get_user_data()

    #AGE
    df['PRE_AGE'] = [34  if x < 34 or np.isnan(x) else x for x in df['Age']]
    df['PRE_AGE'] = [74 if x > 74 else x for x in df['PRE_AGE']]
    df['PRE_AGE'] = [(x-34)/(74-34) for x in df['PRE_AGE']]

    # Sex
    df['PRE_SEX_F'] =   [1 if x=='F' else 0 for x in df['Sex']]
    df['PRE_SEX_M'] =   [1 if x=='M' else 0 for x in df['Sex']]

    # ChestPainType
    df['PRE_CPT_ASY'] =   [1 if x=='ASY'   else 0 for x in df['ChestPainType']]
    df['PRE_CPT_ATA'] =   [1 if x=='ATA'   else 0 for x in df['ChestPainType']]
    df['PRE_CPT_NAP'] =   [1 if x=='NAP'   else 0 for x in df['ChestPainType']]
    df['PRE_CPT_TA'] =   [1 if x=='TA'   else 0 for x in df['ChestPainType']]

    # RestingBP
    df['PRE_RESTINGBP'] = [95  if x < 95 or np.isnan(x) else x for x in df['RestingBP']]
    df['PRE_RESTINGBP'] = [185  if x > 185 else x for x in df['PRE_RESTINGBP']]
    df['PRE_RESTINGBP'] = [(x-95)/(185-95) for x in df['PRE_RESTINGBP']]

    # Cholesterol
    df['PRE_CHOLESTEROL'] = [100  if x < 100 or np.isnan(x) else x for x in df['Cholesterol']]
    df['PRE_CHOLESTEROL'] = [250  if x > 250 else x for x in df['PRE_CHOLESTEROL']]
    df['PRE_CHOLESTEROL'] = [(x-100)/(250-100) for x in df['PRE_CHOLESTEROL']]

    # FastingBS
    df['PRE_FBS_0'] =   [1 if x==0 else 0 for x in df['FastingBS']]
    df['PRE_FBS_1'] =   [1 if x==1 else 0 for x in df['FastingBS']]

    # RestingECG
    df['PRE_RESTECG_LVH'] =   [1 if x=='LVH'   else 0 for x in df['RestingECG']]
    df['PRE_RESTECG_NORMAL'] =   [1 if x=='Normal'   else 0 for x in df['RestingECG']]
    df['PRE_RESTECG_ST'] =   [1 if x=='ST'   else 0 for x in df['RestingECG']]

    # MaxHR
    df['PRE_MAXHR'] = [60  if x < 60 or np.isnan(x) else x for x in df['MaxHR']]
    df['PRE_MAXHR'] = [195  if x > 195 else x for x in df['PRE_MAXHR']]
    df['PRE_MAXHR'] = [(x-60)/(195-60) for x in df['PRE_MAXHR']]

    # ExerciseAngina
    df['PRE_ANGINA_N'] =   [1 if x=='N' else 0 for x in df['ExerciseAngina']]
    df['PRE_ANGINA_Y'] =   [1 if x=='Y' else 0 for x in df['ExerciseAngina']]

    # Oldpeak
    df['PRE_OLDPEAK_ECG'] = [-0.6  if x < -0.6 or np.isnan(x) else x for x in df['Oldpeak']]
    df['PRE_OLDPEAK_ECG'] = [2.4  if x > 2.4 else x for x in df['PRE_OLDPEAK_ECG']]
    df['PRE_OLDPEAK_ECG'] = [(x-(-0.6))/(2.4-(-0.6)) for x in df['PRE_OLDPEAK_ECG']]

    # ST_Slope
    df['PRE_STSLOPE_FLAT'] =   [1 if x=='Flat'   else 0 for x in df['ST_Slope']]
    df['PRE_STSLOPE_UP'] =   [1 if x=='Up'   else 0 for x in df['ST_Slope']]
    df['PRE_STSLOPE_DOWN'] =   [1 if x=='Down'   else 0 for x in df['ST_Slope']]


    #SELECIONANDO VARIAVEIS

    variaveis_selecionadas = ['PRE_AGE','PRE_SEX_F','PRE_SEX_M','PRE_CPT_ASY','PRE_CPT_ATA','PRE_CPT_NAP',
                     'PRE_CPT_TA','PRE_RESTINGBP','PRE_CHOLESTEROL','PRE_FBS_0','PRE_FBS_1','PRE_RESTECG_LVH',
                     'PRE_RESTECG_NORMAL','PRE_RESTECG_ST','PRE_MAXHR','PRE_ANGINA_N','PRE_ANGINA_Y','PRE_OLDPEAK_ECG',
                     'PRE_STSLOPE_FLAT','PRE_STSLOPE_UP','PRE_STSLOPE_DOWN']


    #PREVISÃO
    previsão = modelo.predict(df[variaveis_selecionadas])

    #INSERINDO PREVISÂO NO DF
    df ['Resultado'] = previsão
    
    #SETANDO VARIAVEIS PARA PLOTAR TABELA
    colunas = [
    'Name',
    'Age',
    'Sex',
    'ChestPainType',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'RestingECG',
    'MaxHR',
    'ExerciseAngina',
    'Oldpeak',
    'ST_Slope',
    'Resultado'
        ]

    #SETANDO CODICIONAIS PARA GERAR RESULTADO
    if st.button("Gerar Resultado"):
        if previsão == 1 :
            st.subheader('Resultado - 🚑❤️‍🩹')
            st.write("De acordo com a análise do nosso modelo, há uma probabilidade de ocorrer um evento cardíaco adverso. É importante tomar medidas preventivas imediatas para proteger sua saúde cardiovascular. Recomendamos agendar uma consulta médica o mais breve possível para uma avaliação mais detalhada e elaboração de um plano de cuidados personalizado.")
            st.table(df[colunas])
        else:
            st.subheader('Resultado - 😀🎉')
            st.write('Ficamos felizes em informar que, com base na avaliação do nosso modelo, não há indicação de risco iminente de ataque cardíaco. No entanto, é essencial continuar com hábitos saudáveis, como alimentação balanceada, exercícios regulares e consultas médicas de rotina, para manter sua saúde. Lembre-se de que a prevenção é fundamental para um coração saudável a longo prazo.')
            st.table(df[colunas])







    




