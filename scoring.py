#Возращает оптимальное разбиение непрерывной переменной
def split_numeric(x,y,max_bins):
    x_train_t = np.array(x[x.notnull()]) #Учим только на непустых значениях    
    y_train_t = np.array(y[x.notnull()])
    x_train_t = x_train_t.reshape(x_train_t.shape[0], 1) #Это нужно для работы DecisionTreeClassifier
    m_depth = int(np.log2(max_bins)) + 1 #Максимальная глубина дерева
    bad_rate = y.mean()
    start = 1
    cv_scores = []
    cv = 5
    for i in range(start,m_depth): #Пробегаемся по всем длинам начиная с 1 до максимальной. На каждой итерации делаем кросс-валидацию
        d_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i, min_samples_leaf=0.025)
        scores = cross_val_score(d_tree, x_train_t, y_train_t, cv=cv,scoring='roc_auc')   
        cv_scores.append(scores.mean())
    #    print("Number of bins = ", np.power(2,i),"; GINI = ",2*scores.mean()-1)
    best = np.argmax(cv_scores) + start #Выбираем по максимальному GINI на валидационной выборке
    #print("Optimal number of bins: ", np.power(2,best), "GINI = ",2*max(cv_scores)-1)
    final_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=best, min_samples_leaf=0.025) #Строим финальное дерево
    final_tree.fit(x_train_t, y_train_t)
    #Финальное разбиение
    opt_bins = final_tree.tree_.threshold[final_tree.tree_.feature >= 0]        
    opt_bins = np.append(opt_bins,max(x_train_t)+1)#Добавляем верхнюю границу
    opt_bins = np.append(opt_bins,min(x_train_t)-1)#Добавляем нижнюю границу
    opt_bins = np.sort(opt_bins)    
    return opt_bins #Возвращаем оптимальное разбиение

#Выбирает оптимальное разбиение категориальной переменной
def split_categorial(x,y):
    #One-hot encoding
    x_cat = pd.get_dummies(x,prefix = x.name)
    bad_rate = y.mean()
    max_bins = max(x.nunique(),20)
    #Classification by decision tree
    m_depth = max_bins+1
    start = 1
    cv_scores = []
    cv = 5
    for i in range(start,m_depth):
        d_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i, min_samples_leaf=0.025) 
        scores = cross_val_score(d_tree, x_cat, y, cv=cv,scoring='roc_auc') 
        cv_scores.append(scores.mean())
    #    print("Number of bins = ", i,"; GINI = ",2*scores.mean()-1)
    best = np.argmax(cv_scores) + start #Выбираем по максимальному GINI на валидационной выборке
    #print("Optimal number of bins: ",best, "; GINI = ",2*max(cv_scores)-1)
    final_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=best, min_samples_leaf=0.025) #Строим финальное дерево
    final_tree.fit(x_cat, y)
    
    #Get leafes names
    x_l = final_tree.apply(x_cat)
    tmp = pd.DataFrame(x)
    tmp["LEAF"] = x_l
    
    #Make dictionary with optimal binning
    d = {}
    for leaf in tmp["LEAF"].unique():
        d[leaf]=str(x[tmp["LEAF"]==leaf].unique())   
    tmp["x_num"] = tmp["LEAF"].apply(lambda x: d.get(x))
    return tmp["x_num"]
  
#Пронумеровывает категории по возрастанию
def make_dict(x):        
        x_dict = x.groupby(0)["val"].min().fillna(0).sort_values().reset_index().rename(index=str, columns={0: "x"})
        x_dict['rownum'] = x_dict['val'].rank(method='first', na_option='top')
        x_dict['rownum'] = x_dict['rownum'].apply(zero_pad)
        x_dict['x_num'] = x_dict["rownum"].map(str)+x_dict["x"].map(str)
        del x_dict['val']
        del x_dict['rownum']
        return x_dict    

#Процедура биннинга. 
#Возвращает разбиненную выборку в двух режимах: one-hot или в normal 
#В этих режимах входные данные: 
#      x - выборка в любом формате
#      y - таргеты
#      max_bins - максимальное число групп
#      optimal_bins - для mode = 'normal' или 'one-hot' задает предрассчитанные оптимальные бины
#                     для mode='binning' считает оптимальные бины
#Р
def binning(x,y,max_bins,mode,optimal_bins):
    variable_type = check_type(x)
    if variable_type=='numeric':         
        #Вспомогательная переменная, хранящая разбиения по непустым значениям
        x_bin_t = pd.cut(x[x.notnull()],bins=optimal_bins)    
        #Вспомогательная переменная, хранящая one-hot по непустым значениям
        x_bin = pd.get_dummies(x_bin_t,prefix=x.name,drop_first=True)
        #Добавляем колонку с пустыми значениями
        x_bin[x.name+'_ISNULL']=0
        x_null = pd.DataFrame(x[x.isnull()])
        for i in x_bin.columns:
            x_null[i]=0
        x_null[x.name+'_ISNULL']=1
        del x_null[x.name]
        #Если нет NULL то колонку с dummy is null удаляем   
        if len(x[x.isnull()])==0:
            del x_null[x.name+'_ISNULL']
            del x_bin[x.name+'_ISNULL']
        #Вспомогательная переменная, которая хранит узкий и широкий вид, включая пустые значения    
        x_pivot = pd.concat([x_bin_t,pd.DataFrame(x[x.isnull()])]).sort_index(axis=0)        
        del x_pivot[x.name]
        #Заполняем пустые значения MISSING
        x_pivot = x_pivot.fillna('MISSING')
        x_pivot['val'] = x        
        #Добавляем категориям индекс (создается справочник)
        x_d = make_dict(x_pivot)
        x_pivot["rownum"] = x_pivot.index.values
        x_pivot = pd.merge(x_pivot,x_d,left_on=0,right_on="x").sort_values(by='rownum').reset_index()[["x_num"]]
        #Джойним значения со справочником, удаляем исходные        
        if mode=='one-hot': return pd.concat([x_bin,x_null]).sort_index(axis=0) #Возвращаем в виде on-hot                            
        if mode=='normal': return x_pivot #Возвращаем в "длинном и узком" виде               
    if variable_type=='cat': 
        x_bin = split_categorial(x,y)          
        if mode=='one-hot': return pd.get_dummies(x_bin,prefix=x.name,drop_first=True)
        if mode=='normal': return pd.DataFrame(x_bin)
    if (mode=='binning')&(variable_type=='numeric'):
        x_bins = split_numeric(x,y,max_bins)
        return x_bins
        
#Добавляет лидирующие нули к категориям          
def zero_pad(x):
    if str(x)=='MISSING': return '000'
    if len(str(x))==3: return str('00'+str(x))[:-2]+': '
    if len(str(x))==4: return str('0'+str(x))[:-2]+': '
    if len(str(x))==5: str(x)[:-2]+': '

#Считает Information Value, Weight of evidence для заданного разбиения       
def iv_table(x,y):
    #На вход подается разбиненная с помощью процедуры binning переменная - x
    #y - целевая переменная (флаги дефолта)
    df_t = x
    df_t["y"] = y
    df_t = df_t.rename(index=str, columns = {"x_num":"x"})
    df_iv =pd.DataFrame({'count': df_t.groupby('x')['y'].count(), 
                     'bad_rate': df_t.groupby('x')['y'].mean(),
                     'total_goods': df_t.groupby('x')['y'].count() - df_t.groupby('x')['y'].sum(),
                     'total_bads': df_t.groupby('x')['y'].sum() 
                     }).reset_index()
    df_iv["cumm_bads"] = df_iv['total_bads'].cumsum()
    df_iv["cumm_goods"] = df_iv['total_goods'].cumsum()
    df_iv["cumm_total"] = df_iv['count'].cumsum()
    df_iv["per_bad"] = df_iv["total_bads"]/df_iv["cumm_bads"].max()
    df_iv["per_good"] = df_iv["total_goods"]/df_iv["cumm_goods"].max()
    df_iv["woe"] = np.log((df_iv["per_good"])/(df_iv["per_bad"]+0.000000001))
    iv = (df_iv["per_good"] - df_iv["per_bad"])*np.log((df_iv["per_good"])/(df_iv["per_bad"]+0.000000001))
    df_iv["iv"] = iv.sum()       
    return df_iv
    
#Выводит IV по переменной. На вход принимает данные в формате iv_table    
def iv_value (df_iv):
    return df_iv["iv"].mean()

#На вход подается массив, на выходе - признак: числовой или категориальный
def check_type(x):
    from pandas.api.types import is_string_dtype
    from pandas.api.types import is_numeric_dtype   
    #Удаляем пустые значения
    x = x[x.notnull()]
    #Если число различных значений меньше 4, то тип-категориальный
    if x.nunique()<=4: return 'cat'
    elif is_numeric_dtype(x): return 'numeric'
    else: return 'cat'

#Процедура отбора переменных по IV. На вход принимает список переменных, на выход выдает отчетные таблицы и оптимальное разбиение   
def iv_selection(x,y,iv_threshold):
    print("Choosing variables with IV > ",iv_threshold)
    var_list = []
    #Структура для хранения оптимального разбиения
    x_bins = {'name': [1,2,3]}
    for i in x.columns:
        x_bins[i] = binning(x[i],y,max_bins=8,mode='binning',optimal_bins=1)
        x_bin = binning(x[i],y,max_bins=8,mode='normal',optimal_bins=x_bins[i])
        x_iv = iv_table(x_bin,y)
        iv = iv_value(x_iv)
        if (iv<iv_threshold)|(iv>5): x_bins.pop(i)
        print('________________________________________________________________________________________________________________')
        print('                                                             ')
        print(i,"  IV = ", iv)
        print(x_iv)
    x_bins.pop('name')
    return x_bins    

#Процедура преобразования выборки в one-hot, учитывая биннинг. Нужно для подачи на вход процедуры расчета корреляций
def dev_to_one_hot(x,y,list_optimal_bins):    
    x_dev = pd.DataFrame(x.index.values)
    for i in x.columns:
        x_bin = binning(x[i],y,max_bins=8,mode='one-hot',optimal_bins = list_optimal_bins[i])
        x_dev = pd.merge(x_dev,x_bin,left_index=True,right_index=True)
    del x_dev[0]
    return x_dev

#Проверка, если количество различных категорий велико (Id-шники, даты, и т.д.) для того, чтобы выкинуть эти колонки
def check_mass_cat(x):
    drop_list=[]
    for i in range(len(x.columns)):
        x[x.columns[i]] = x[x.columns[i]].fillna(0)
        #Если количество уникальных значений >= количеству строк / 2 и тип - категориальный
        if x[x.columns[i]].nunique()>len(x)/2 and check_type(x[x.columns[i]])=='cat': drop_list.append(x.columns[i])
        #Если на самую крупную группу приходится менее 1% выборки    
        if max(x.groupby(x.columns[i])[x.columns[0]].count())/len(x)<0.01 and check_type(x[x.columns[i]])=='cat': drop_list.append(x.columns[i])
    #Конец формирования списка переменных, которые надо выкинуть
    #Формируем список переменных, которые надо оставить
    var_list = x.columns.values
    final_list=[]
    for i in range(len(x.columns)):
        x[x.columns[i]] = x[x.columns[i]].fillna(0)
        #Если количество уникальных значений >= количеству строк / 2 и тип - категориальный
        if x[x.columns[i]].nunique()>len(x)/3 and check_type(x[x.columns[i]])=='cat': drop_list.append(x.columns[i])
        #Если на самую крупную группу приходится менее 1% выборки    
        if max(x.groupby(x.columns[i])[x.columns[0]].count())/len(x)<0.01 and check_type(x[x.columns[i]])=='cat': drop_list.append(x.columns[i])
    for elem in var_list:
        if elem not in drop_list: final_list.append(elem)
    return x[final_list]

#Принимает на вход выборку в виде one-hot, на выходе дает ту же выборку с исключенными коррелирующими факторами
def exclude_corr_factors(x_dev_t, corr_threshold,mode):
    x_corr = x_dev_t.corr()
    #Оставляем только колонки - потенциальные кандидаты на исключение (хотя бы одно значение корреляции выше трешхолда)
    col_list=[]    
    for i in range(len(x_corr.columns)):
        #Заменяем диагональные значения на 0    
        x_corr[x_corr.columns[i]][x_corr[x_corr.columns[i]].index.values[i]] = 0
        #Если в колонке найдено, хотя бы одно значение с корреляцией больше трешхолда, добавляем ее в лист
        if max(abs(x_corr[x_corr.columns[i]]))>corr_threshold: col_list.append(x_corr.columns[i])
    #Оставляем только те колонки, из которых нужно выбрать которые выкинуть из-за корреляций            
    x_dev_drop =  x_dev_t[col_list]
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
        if max(abs(x_c[x_c.columns[0]]))>=corr_threshold:     
            corr_list[exclude_iteration].append(x_c.columns[0])
        #Пробегаемся по всем колонкам
            for i in range(len(x_c.columns)):
        #Записываем в пару к первой все коррелирующие с ней колонки
                if abs(x_c[x_c.columns[0]].iloc[i])>=corr_threshold:
                    corr_list[exclude_iteration].append(x_c.columns[i])
            #Выкидываем все колонки, которые коррелируют с первой
            var_list = [x for x in x_c.columns.values if x not in corr_list[exclude_iteration]]
            x_dev_drop = x_dev_drop[var_list]
            x_c = x_dev_drop.corr()
            corr_list.append([])
            exclude_iteration = exclude_iteration+1
            print("Excluding correlations. Iteration = ",exclude_iteration,"Corr list: ", corr_list)
    #После обработки corr_list содержит все списки коррелирующих колонок. Из каждого списка оставляем только одну
    cols_to_drop=[] #Список колонок, которые надо выкинуть
    for i in range(len(corr_list)):
        for j in range(len(corr_list[i])):
            if j!=0: 
                cols_to_drop.append(corr_list[i][j])
    #Оставляем в исходном списке только колонки не из col_to_drop
    exclude_list = [x for x in x_dev_t.columns.values if x not in cols_to_drop]
    x_dev_t = x_dev_t[exclude_list]
    if mode=='var': return x_dev_t
    if mode=='list': return exclude_list

#Строит скоркарту
def build_model(x,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logit_model = LogisticRegression()
    logit_model.fit(x,y)
    return logit_model

#Выводит готовую скоркарту
def scorecard_view(col_list, model, odds_X_to_one,odds_score,double_odds):
    print('Printing scorecard...')
    cols = np.array('Intercept')
    cols = np.append(cols,np.array(col_list))
    vals = np.array(model_logit.intercept_)
    vals = np.append(vals,np.array(model_logit.coef_))
    scorecard = pd.DataFrame(cols)
    scorecard.rename(columns={0: 'Variable'},inplace=True)
    scorecard["Regression_coef"] = pd.DataFrame(vals)
    b = double_odds/np.log(2)
    a = odds_score - b*np.log(odds_X_to_one)    
    scorecard["Score"] = scorecard["Regression_coef"]*b
    scorecard["Score"][0] = scorecard["Score"][0]+a
    scorecard["Score"] = round(scorecard["Score"],2)
    return scorecard

def gini(model,x,y):
    print('GINI = ',2*roc_auc_score(y,model.predict_proba(x)[:,1])-1)
            