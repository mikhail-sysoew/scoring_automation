import sklearn as sk
import numpy as np
import pandas as pd
import warnings
import time
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import model_selection
from sklearn import cross_validation
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.simplefilter('ignore')


class Scorecard():
    def __init__(self, max_bins=8, minimum_leaf=0.025, corr_threshold=0.8, odds_X_to_one = 100, odds_score=700, double_odds=25):        
        self.regressor=LogisticRegression() #Regression build method
        self.x = pd.DataFrame() #Input sample
        self.y = pd.DataFrame() #Targets
        self.vars = []
        self.vars_after_iv_cut = []
        self.vars_after_corr_cut = []
        self.var_list_types = {} #Types of variables
        self.var_list_bins = {} #Binning of scorecard variables dictionary
        self.scorecard = pd.DataFrame() #Final scorecard representation
        self.iv_table = {} #information value tables for each variable
        self.gini = int #Gini of model 
        self.logit_model = [] #model object for LogisticRegression
        self.max_bins = max_bins #Regularization parameter. Maximum bins used in decision tree
        self.minimum_leaf = minimum_leaf #Regularization parameter. Mininmum size of one leaf
        self.column = ''
        self.iv_table = {} #Dictionary which contains iv table for each variable
        self.x_one_hot = pd.DataFrame() #Input sample in one-hot view
        self.corr_threshold = corr_threshold
        self.odds_X_to_one = odds_X_to_one 
        self.odds_score = odds_score
        self.double_odds = double_odds
        self.x_binned = pd.DataFrame()
        
   
  
    #Learn model on sample
    def fit(self,x,y,iv_treshold):        
        self.x = x
        self.y = y
        self.x = self.x.reset_index()
        self.y = self.y.reset_index()
        del self.x["index"]
        del self.y["index"]      
        self.fill_vars_cats()          
        #print('Binning columns...')
        #fill all values of var_list_bins
        for col in self.x.columns: 
            #print(col)
            self.binning(mode_forward='binning',mode_output='normal',column_name=col)  
            #Filling IV table on current variable
            df_t = pd.DataFrame(self.binning(mode_forward='forward',mode_output='normal',column_name=col))
            df_t["y"] = self.y
            #df_t = df_t.rename(index=str, columns = {col:"x"})
            df_iv =pd.DataFrame({'count': df_t.groupby(col)['y'].count(), 
                             'bad_rate': df_t.groupby(col)['y'].mean(),
                             'total_goods': df_t.groupby(col)['y'].count() - df_t.groupby(col)['y'].sum(),
                            'total_bads': df_t.groupby(col)['y'].sum() 
                             }).reset_index()
            df_iv["cumm_bads"] = df_iv['total_bads'].cumsum()
            df_iv["cumm_goods"] = df_iv['total_goods'].cumsum()
            df_iv["cumm_total"] = df_iv['count'].cumsum()
            df_iv["per_bad"] = df_iv["total_bads"]/df_iv["cumm_bads"].max()
            df_iv["per_good"] = df_iv["total_goods"]/df_iv["cumm_goods"].max()
            df_iv["woe"] = np.log((df_iv["per_good"])/(df_iv["per_bad"]+0.000000001))
            iv = (df_iv["per_good"] - df_iv["per_bad"])*np.log((df_iv["per_good"])/(df_iv["per_bad"]+0.000000001))
            df_iv["iv"] = iv.sum()       
            self.iv_table[col] = df_iv
            if df_iv["iv"].mean()>=iv_treshold: self.vars_after_iv_cut.append(col)
        #creating sample in one-hot view
        self.x_one_hot = pd.DataFrame(self.x.index.values)       
        for col in self.vars_after_iv_cut:          
            self.x_one_hot = pd.merge(self.x_one_hot, pd.DataFrame(self.binning(mode_forward='forward',mode_output='one-hot',column_name=col)),left_index=True,right_index=True)
        del self.x_one_hot[self.x_one_hot.columns[0]]
        self.x = self.x[self.vars_after_iv_cut] 
        #print('Exclude correlations...')
        self.exclude_corr_factors()   
        #print('Building regression...')
        self.regressor.fit(self.x_one_hot,self.y)
        self.scorecard_view()
        
    def predict_proba(self,x):
        self.x = x        
        self.x = x.reset_index()
        del self.x["index"]
        self.x_binned = pd.DataFrame(self.x.index.values)
        cols_to_delete = set(self.x.columns) - set(self.vars_after_iv_cut)
        for c in cols_to_delete:
            del self.x[c]
        for col in self.x.columns:
            self.x_binned = pd.merge(self.x_binned,pd.DataFrame(self.binning(mode_forward='forward',mode_output='one-hot',column_name = col)),left_index=True,right_index=True)
            #del x_binned[x_binned.columns[0]]
        cols_to_delete = set(self.x_binned.columns) - set(self.scorecard["Variable"])
        for c in cols_to_delete:
            del self.x_binned[c]
        return self.regressor.predict_proba(self.x_binned)[:,1]
        
    def predict_score(self,x):
        y_pred = self.predict_proba(x)
        bias = self.odds_score - self.double_odds*np.log(self.odds_X_to_one)/np.log(2)   
        odds = self.double_odds/np.log(2)         
        return bias+odds*np.log(1/y_pred-1)  
      
    
    def scorecard_view(self):
      #  print('Printing scorecard...')
        self.scorecard=[]
        cols = np.array('Intercept')
        cols = np.append(cols,np.array(self.vars_after_corr_cut))
        vals = np.array(self.regressor.intercept_)
        vals = np.append(vals,np.array(self.regressor.coef_))
        self.scorecard = pd.DataFrame(cols)
        self.scorecard.rename(columns={0: 'Variable'},inplace=True)
        self.scorecard["Regression_coef"] = pd.DataFrame(vals)
        b = self.double_odds/np.log(2)
        a = self.odds_score - b*np.log(self.odds_X_to_one)    
        self.scorecard["Score"] = self.scorecard["Regression_coef"]*b
        self.scorecard["Score"][0] = self.scorecard["Score"][0]+a
        self.scorecard["Score"] = round(self.scorecard["Score"],2)
        
    
    
    #Exclude correlations. Fill vars_after_corr_cut. Exclude correlated columns from x_one_hot
    def exclude_corr_factors(self):
        x_corr = self.x_one_hot.corr()
        #Оставляем только колонки - потенциальные кандидаты на исключение (хотя бы одно значение корреляции выше трешхолда)
        col_list=[]    
        for i in range(len(x_corr.columns)):
            #Заменяем диагональные значения на 0    
            x_corr[x_corr.columns[i]][x_corr[x_corr.columns[i]].index.values[i]] = 0
            #Если в колонке найдено, хотя бы одно значение с корреляцией больше трешхолда, добавляем ее в лист
            if max(abs(x_corr[x_corr.columns[i]]))>self.corr_threshold: col_list.append(x_corr.columns[i])
        #Оставляем только те колонки, из которых нужно выбрать которые выкинуть из-за корреляций            
        x_dev_drop =  self.x_one_hot[col_list]
        #Строим корреляционную матрицу из оставшихся
        x_c = x_dev_drop.corr()
        #Пустой список
        corr_list = []
        corr_list.append([])
        exclude_iteration = 0
        var_list = [0,1]
        #Заполняем диагональ нулями
        for i in range(len(x_c.columns)):        
            x_c[x_c.columns[i]][x_c[x_c.columns[i]].index.values[i]] = 0
        while len(var_list)>1&len(x_c)>0:
            for i in range(len(x_c.columns)):        
                x_c[x_c.columns[i]][x_c[x_c.columns[i]].index.values[i]] = 0
            #Если нашли хотя бы одну колонку, которая коррелирует с первой, создаем пару в corr_list и записываем туда первую колонку
            if max(abs(x_c[x_c.columns[0]]))>=self.corr_threshold:     
                corr_list[exclude_iteration].append(x_c.columns[0])
            #Пробегаемся по всем колонкам
                for i in range(len(x_c.columns)):
            #Записываем в пару к первой все коррелирующие с ней колонки
                    if abs(x_c[x_c.columns[0]].iloc[i])>=self.corr_threshold:
                        corr_list[exclude_iteration].append(x_c.columns[i])
                #Выкидываем все колонки, которые коррелируют с первой
                var_list = [x for x in x_c.columns.values if x not in corr_list[exclude_iteration]]
                x_dev_drop = x_dev_drop[var_list]
                x_c = x_dev_drop.corr()
                corr_list.append([])
                exclude_iteration = exclude_iteration+1
                #print("Excluding correlations. Iteration = ",exclude_iteration,"Corr list: ", corr_list)
        #После обработки corr_list содержит все списки коррелирующих колонок. Из каждого списка оставляем только одну
        cols_to_drop=[] #Список колонок, которые надо выкинуть
        for i in range(len(corr_list)):
            for j in range(len(corr_list[i])):
                if j!=0: 
                    cols_to_drop.append(corr_list[i][j])
        #Оставляем в исходном списке только колонки не из col_to_drop
        exclude_list = [x for x in self.x_one_hot.columns.values if x not in cols_to_drop]
        self.x_one_hot = self.x_one_hot[exclude_list]
        self.vars_after_corr_cut = exclude_list
        
    
    #Input - one variable name 
    #Output - optimal binning, builded on decision tree. Maximum number of bins = max_bins
    def split_numeric(self,column_name):  
        x_train_t = np.array(self.x[column_name][self.x[column_name].notnull()]) #Exclude nulls 
        y_train_t = np.array(self.y[self.x[column_name].notnull()])
        x_train_t = x_train_t.reshape(x_train_t.shape[0], 1) #Need for DecisionTreeClassifier
        m_depth = int(np.log2(self.max_bins)) + 1 #Maximum tree depth
        #bad_rate = self.y.mean()
        start = 1
        cv_scores = []
        cv = 3
        for i in range(start,m_depth): #Loop over all tree depth. CV on the each step
            d_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i, min_samples_leaf=self.minimum_leaf)
            scores = cross_val_score(d_tree, x_train_t, y_train_t, cv=cv,scoring='roc_auc')   
            cv_scores.append(scores.mean())        
        best = np.argmax(cv_scores) + start #Criterion - maximum GINI on validation set        
        final_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=best, min_samples_leaf=0.025) #Build final tree
        final_tree.fit(x_train_t, y_train_t)
        #Final tree
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.feature >= 0]        
        opt_bins = np.append(opt_bins,max(x_train_t)+1)#Add right border
        opt_bins = np.append(opt_bins,min(x_train_t)-1)#Add left border
        opt_bins = np.sort(opt_bins)    
        return opt_bins #Return optimal binning
    
    #Split categorial variable. Grouping variable for regularization.
    #Input = column name
    #Output : add to var_list_bins binned variable as dictionary
    def split_categorial(self,column_name):
        #One-hot encoding
        self.x[column_name] = self.x[column_name].fillna('MISSING')
        x_cat = pd.get_dummies(self.x[column_name],prefix = self.x[column_name].name)
        #bad_rate = self.y.mean()
        max_bins = max(self.x[column_name].nunique(),20)
        #Classification by decision tree
        m_depth = max_bins+1
        start = 1
        cv_scores = []
        cv = 3
        for i in range(start,m_depth):
            d_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i, min_samples_leaf=self.minimum_leaf) 
            scores = cross_val_score(d_tree, x_cat, self.y, cv=cv,scoring='roc_auc') 
            cv_scores.append(scores.mean())
        #    print("Number of bins = ", i,"; GINI = ",2*scores.mean()-1)
        best = np.argmax(cv_scores) + start #Choose maximizing GINI on validation dataset
        #print("Optimal number of bins: ",best, "; GINI = ",2*max(cv_scores)-1)
        final_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=best, min_samples_leaf=0.025) #Build final tree
        final_tree.fit(x_cat, self.y)

        #Get leafes names
        x_l = final_tree.apply(x_cat)
        tmp = pd.DataFrame(self.x[column_name])
        tmp["LEAF"] = x_l

        #Make dictionary with optimal binning
        d = {}
        for leaf in tmp["LEAF"].unique():
            d[leaf]=str(self.x[column_name][tmp["LEAF"]==leaf].unique())   
        tmp["x_num"] = tmp["LEAF"].apply(lambda x: d.get(x))
        return d
   
    #Define variable category - numeric or categorial
    #Input - column name
    #Output - numeric or cat
    def check_type(self,column_name):
        from pandas.api.types import is_string_dtype
        from pandas.api.types import is_numeric_dtype   
        #delete nulls
        tmp_var = self.x[column_name][self.x[column_name].notnull()]
        #If number of uniques<=4 then type = categorial
        if tmp_var.nunique()<=4: return 'cat'
        elif is_numeric_dtype(tmp_var): return 'numeric'
        else: return 'cat'
    
    #Fill variable var_list_cats
    def fill_vars_cats(self):
        from pandas.api.types import is_string_dtype
        from pandas.api.types import is_numeric_dtype 
        for col in self.x[self.x.columns]:
            if self.check_type(col)=='numeric': self.var_list_types[col]='numeric'
            if self.check_type(col)=='cat': 
                self.var_list_types[col]='cat'
                if (self.x[col].nunique()<=4)&(is_numeric_dtype(self.x[col])): self.x[col] = self.x[col].apply(lambda x: 'cat_'+str(x))
                
    
    #Add leading zeros to names
    def zero_pad(self,x):
        if str(x)=='MISSING': return '000'
        if len(str(x))==3: return str('00'+str(x))[:-2]+': '
        if len(str(x))==4: return str('0'+str(x))[:-2]+': '
        if len(str(x))==5: str(x)[:-2]+': '
    
    #Naming for categories by rank
    def make_dict(x):        
        x_dict = x.groupby(0)["val"].min().fillna(0).sort_values().reset_index().rename(index=str, columns={0: "x"})
        x_dict['rownum'] = x_dict['val'].rank(method='first', na_option='top')
        x_dict['rownum'] = x_dict['rownum'].apply(zero_pad)
        x_dict['x_num'] = x_dict["rownum"].map(str)+x_dict["x"].map(str)
        del x_dict['val']
        del x_dict['rownum']
        return x_dict   
    
    #Binning procedure
    #Return binned sample. Has two modes - one-hot and norma;
    #Inputs 
    #      x - sample
    #      y - targets
    #      max_bins - maximum number of bins
    #      optimal_bins - for mode_output = 'normal' or 'one-hot' using as input for feed forward
    #                     for mode_forward='binning' calculating of optimal bins
    #                         mode_forward='forward' calculating outputs using optimals bins as input 
    #
    
    #Need for feed forward categorial variables
    #Take value from dictionary var_list_bins and answer if current value is in list
    #If yes - return list
    
    
    def forward_cat(self,x):
        for i in self.var_list_bins[self.column].keys():
            if str(x) in self.var_list_bins[self.column][i]:
                return str(self.var_list_bins[self.column][i]) 
    
    def binning(self,mode_output,mode_forward,column_name):
        variable_type = self.var_list_types[column_name]
        if (variable_type=='numeric')&(mode_forward=='forward'):         
            #Вспомогательная переменная, хранящая разбиения по непустым значениям
            x_bin_t = pd.cut(self.x[column_name][self.x[column_name].notnull()],bins=self.var_list_bins[column_name])    
            #Вспомогательная переменная, хранящая one-hot по непустым значениям
            x_bin = pd.get_dummies(x_bin_t,prefix=self.x[column_name].name,drop_first=True)
            #Добавляем колонку с пустыми значениями
            x_bin[self.x[column_name].name+'_ISNULL']=0
            x_null = pd.DataFrame(self.x[column_name][self.x[column_name].isnull()])
            for i in x_bin.columns:
                x_null[i]=0
            x_null[self.x[column_name].name+'_ISNULL']=1
            del x_null[self.x[column_name].name]
            #Если нет NULL то колонку с dummy is null удаляем   
            if len(self.x[column_name][self.x[column_name].isnull()])==0:
                del x_null[self.x[column_name].name+'_ISNULL']
                del x_bin[self.x[column_name].name+'_ISNULL']
            #Вспомогательная переменная, которая хранит узкий и широкий вид, включая пустые значения    
            x_pivot = pd.concat([x_bin_t,pd.DataFrame(self.x[column_name][self.x[column_name].isnull()])]).sort_index(axis=0)        
            del x_pivot[self.x[column_name].name]
            #Заполняем пустые значения MISSING
            x_pivot = x_pivot.fillna('MISSING')
            x_pivot['val'] = self.x[column_name]        
            #Добавляем категориям индекс (создается справочник)           
            x_dict = x_pivot.groupby(0)["val"].min().fillna(0).sort_values().reset_index().rename(index=str, columns={0: "x"})
            x_dict['rownum'] = x_dict['val'].rank(method='first', na_option='top')
            x_dict['rownum'] = x_dict['rownum'].apply(self.zero_pad)
            x_dict[column_name] = x_dict["rownum"].map(str)+x_dict["x"].map(str)
            del x_dict['val']
            del x_dict['rownum']
            x_d =  x_dict   
            x_pivot["rownum"] = x_pivot.index.values
            x_pivot = pd.merge(x_pivot,x_d,left_on=0,right_on="x").sort_values(by='rownum').reset_index()[column_name]
            #Джойним значения со справочником, удаляем исходные        
            if mode_output=='one-hot': return pd.concat([x_bin,x_null]).sort_index(axis=0) #Возвращаем в виде on-hot                            
            if mode_output=='normal': return x_pivot #Возвращаем в "длинном и узком" виде               
        if (variable_type=='cat')&(mode_forward=='forward'): 
            ####################INPUT CODE HERE#####################
            if mode_output=='normal': 
                self.column = column_name
                return self.x[column_name].apply(self.forward_cat)
            if mode_output=='one-hot': 
                self.column = column_name
                return pd.get_dummies(self.x[column_name].apply(self.forward_cat), drop_first=True,prefix=self.x[column_name].name)
        if (variable_type=='numeric')&(mode_forward=='binning'):
            self.var_list_bins[column_name] = self.split_numeric(column_name)
        if (variable_type=='cat')&(mode_forward=='binning'):                
            self.var_list_bins[column_name] = self.split_categorial(column_name)
            #x_bin = self.split_categorial(column_name)          
            #if mode_output=='one-hot': return pd.get_dummies(x_bin,prefix=self.x[column_name].name,drop_first=True)
            #if mode_output=='normal': return pd.DataFrame(x_bin)