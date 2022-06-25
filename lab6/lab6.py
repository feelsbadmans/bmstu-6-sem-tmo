import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TARGET_COL_NAME = 'hazardous'

# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
clas_models = {'LogR': LogisticRegression(),
               'KNN_5': KNeighborsClassifier(n_neighbors=5),
               'SVC': SVC(probability=True),
               'Tree': DecisionTreeClassifier(),
               'RF': RandomForestClassifier(),
               'GB': GradientBoostingClassifier()}

# Загрузка датасета
def load_data():
    df = pd.read_csv('lab6/data/neo.csv')
    sub_df = df.drop(['id','name','orbiting_body','sentry_object'],axis=1)
    sub_df['hazardous'] = [0 if entry==False else 1 for entry in sub_df['hazardous']]
    sub_df = sub_df.iloc[0:10000]

    return sub_df

# Масштабирование данных и разделение выборки
def preprocess_data(df_in):
    df_out = df_in.copy()
    df_out = df_out.dropna(axis=1, how='any')

    TARGET_IS_NUMERIC = df_out[TARGET_COL_NAME].dtype != 'O'
    number_cols = df_out.select_dtypes(exclude=['object'])
    not_number_cols = df_out.select_dtypes(include=['object'])

    le = LabelEncoder()
    scaler = MinMaxScaler()

    for col_name in not_number_cols:
        df_out[col_name] = le.fit_transform(df_out[col_name])

    number_fields_source = number_cols.loc[:, number_cols.columns !=
                                           TARGET_COL_NAME] if TARGET_IS_NUMERIC else number_cols

    for col_name in number_fields_source:
        df_out[col_name] = scaler.fit_transform(df_out[[col_name]])

    df_x = df_out.loc[:, df_out.columns != TARGET_COL_NAME]
    df_y = df_out[TARGET_COL_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test


# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(
        y_true, y_score, average=average, multi_class="ovo")
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

# 
def print_models(models_select, X_train, X_test, y_train, y_test):
    current_models_list = []
    roc_auc_list = []
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)

        Y_pred = model.predict(X_test)
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:, 1]

        roc_auc = roc_auc_score(y_test.values, Y_pred_proba, multi_class="ovo")
        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        draw_roc_curve(y_test.values, Y_pred_proba, ax[0], )
        cm = confusion_matrix(
            y_test, Y_pred, normalize='all', labels=model.classes_)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax[1], cmap=plt.cm.Blues)
        fig.suptitle(model_name)
        st.pyplot(fig)

    if len(roc_auc_list) > 0:
        temp_d = {'roc-auc': roc_auc_list}
        temp_df = pd.DataFrame(data=temp_d, index=current_models_list)
        st.bar_chart(temp_df)


st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)

st.header('Оценка качества моделей')
print_models(models_select, X_train, X_test, y_train, y_test)
