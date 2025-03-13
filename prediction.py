import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
train_extra = pd.read_csv("training_extra.csv")
test = pd.read_csv("test.csv")

train.shape, test.shape, train_extra.shape

# concat the train data and extra data
train = pd.concat([train, train_extra], ignore_index=True)
train.shape

train.head()


# Eksik değerleri doldurmak, kategorik değişkenleri sayısal hale getirmek ve yeni özellikler mühendisliği (feature engineering) yaparak veri setini modellemeye hazır hale getirmek.
dict_fen = {'Material':'NaN','Style':'NaN','Brand':'NaN','Size':'NaN','Waterproof':'NaN','Color':'NaN','Laptop Compartment':'NaN'}

def feh(df):
    df = df.fillna(dict_fen)
    map_size = {'Small':1.1, 'Medium':1.2, 'Large':1.3, 'NaN':0}
    map_brand      = {'Jansport': 1.1,'Adidas':1.2,'Nike':1.3,'Puma':1.4,'Under Armour':1.5,'NaN':0}
    map_color      = {'Black':1.1,'Green':1.2,'Red':1.3,'Blue':1.4,'Gray':1.05,'Pink':1.5,'NaN':0}
    map_style      = {'Messenger':1.1,'Backpack':1.2,'Tote': 1.3,'NaN':0}
    map_material   = {'Polyester':1.1,'Leather': 1.2,'Nylon':1.3,'Canvas':1.4,'NaN':0}
    map_waterproof = {'Yes':1.1,'No':1.0,'NaN':0}
    map_laptop     = {'Yes':1.1,'No':1.0,'NaN':0}

    df['Size_map'] = df['Size'].map(map_size)
    df['Brand_map'] = df['Brand'].map(map_brand)
    df['Color_map'] = df['Color'].map(map_color)
    df['Style_map'] = df['Style'].map(map_style)
    df['Material_map'] = df['Material'].map(map_material)
    df['Waterproof_map'] = df['Waterproof'].map(map_waterproof)
    df['Laptop_map'] = df['Laptop Compartment'].map(map_laptop)
    df['Compartments_map'] = df['Compartments'].apply(lambda x: x/1.1)
    
    df['_NaN_Material'] = df['Material'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Style'] = df['Style'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Size'] = df['Size'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Waterproof'] = df['Waterproof'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Color'] = df['Color'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Laptop'] = df['Laptop Compartment'].apply(lambda x: 1 if x == 'NaN' else 0)

    df['_7_NaNs'] = df['_NaN_Material'] + df['_NaN_Style'] + df['_NaN_Brand'] + df['_NaN_Size'] + df['_NaN_Waterproof'] + df['_NaN_Color'] + df['_NaN_Laptop']

    df = df.rename(columns={'Size_map':'x1','Brand_map':'x2','Color_map':'x3','Style_map':'x4','Material_map':'x5','Waterproof_map':'x6','Laptop_map':'x7','Compartments_map':'x8'})

    median_weight = df['Weight Capacity (kg)'].median()
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(median_weight)

    conditions = [
        (df['Weight Capacity (kg)'] <= 5),
        (df['Weight Capacity (kg)'] > 5) & (df['Weight Capacity (kg)'] <= 15),
        (df['Weight Capacity (kg)'] > 15) & (df['Weight Capacity (kg)'] <= 20),
        (df['Weight Capacity (kg)'] > 20) & (df['Weight Capacity (kg)'] <= 25),
        (df["Weight Capacity (kg)"] > 25)
    ]

    choices = ['Light', 'Middle', 'Light_heavy', 'Middel_heavy', 'Heavy']
    df['Weight_Class'] = np.select(conditions, choices, default='')
    df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].astype(float)
    df['Weight_Class'] = df['Weight_Class'].astype('category')

    return df

train = feh(train)
test = feh(test)

train.info()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(train.drop(columns='Price'),train['Price'],test_size=0.2)

#  object (string) veri tipindeki sütunları category tipine çevirmek için kullanılıyor.
from sklearn.base import TransformerMixin
class ObjectToCategory(TransformerMixin):
    def fit(self, X, y=None): # fit methodu sadece sütun türlerini öğrenir ve self(kendini) döndürür
        return self
    def transform(self, X):
        X = pd.DataFrame(X) # Gelen veriyi bir pandas DataFrame formatına çeviriyoruz.
        self.categorical_columns_ = X.select_dtypes(include=['object']).columns.tolist() # metin içeren sütunları seçiyoruz.
        for col in self.categorical_columns_:
            X[col] = X[col].astype('category') # Listede bulunan her sütunu category türüne çeviriyoruz.
        return X
    
obj_to_cat = ObjectToCategory()
X_train_transformed = obj_to_cat.fit_transform(X_train)

categorical_columns = obj_to_cat.categorical_columns_

X_train.info()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    num_leaves=56,
    learning_rate=0.022497950121387757,
    n_estimators=1400,
    min_child_samples=419,
    subsample=0.9318779151947196,
    colsample_bytree=0.5935633028324053,
    reg_alpha=1.0644656664600252,
    reg_lambda=0.3945627132333395,
    min_split_gain=9.98148173286267e-07,
    max_bin=1899,
    min_data_in_leaf=403,
    cat_features=categorical_columns
    )

rmse_scorer = make_scorer(mean_squared_error, squared=False)
scores = cross_val_score(model, X_train_transformed, y_train, cv=5, scoring=rmse_scorer)
print(f'Cross-validated RMSE: {np.mean(scores): .4f}')

model.fit(X_train_transformed, y_train)


X_test_transformed = obj_to_cat.transform(X_test)
y_pred = model.predict(X_test_transformed)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test Set RMSE: {test_rmse: .4f}')

X_test_transformed = obj_to_cat.transform(test)
predictions = model.predict(X_test_transformed)

submission = pd.DataFrame({'id': test['id'], 'Price': predictions})

submission.to_csv('submission1.csv', index=False)

# Bu kod şuan en iyi skoru veriyor.
# Test Set RMSE:  38.8432