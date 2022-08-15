from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import statistics as st

#im = plt.imread('C:\\Users\\Zixu.Liu\\AppData\\LocalLow\\DefaultCompany\\screenshotsSizeTest13\\screen_512x512_26.png')

im = plt.imread('C:\\Users\\Zixu.Liu\\AppData\\LocalLow\\DefaultCompany\\apple30\\screen_512x512_20.png')
with open('C:\\Users\\Zixu.Liu\\AppData\\LocalLow\\DefaultCompany\\apple30\\20.txt') as f:
    label = json.load(f)
labelClass='_apple'
#xs = [point['x']*100 for point in label if point['label'] == labelClass]
#ys = [point['y']*100 for point in label if point['label'] == labelClass]

#xs = [point['x'] * 100 for point in label if labelClass in point['label']]    ###after
#ys = [point['y'] * 100 for point in label if labelClass in point['label']]    ###after

xs = [point['x'] for point in label ]
ys = [point['y'] for point in label ]

#plt.imshow(im)
#plt.plot(xs,ys,'bo')
#plt.show()

##fff=open('C:\\Users\\Zixu.Liu\\AppData\\LocalLow\\DefaultCompany\\screenshotsTest4\\2.txt', "r")
##label = json.load(fff)
##xxxx=list(zip(xs, ys))
##my_array = np.asarray(xxxx)
#y=np.arange(0,128)
#x=np.arange(0,128)


#fig, ax = plt.subplots()
#cs=ax.pcolormesh(x, y, annots['densityArray'][0][0][x][y])
#fig.colorbar(cs)

##ax.invert_yaxis()

##from scipy.io import loadmat
## annots = loadmat('C:\\Work\\qianproject\\test result\\headCount_vgg16_bn RMSprop\\results_headCount_vgg16_bn_shA_RMSprop.mat')

# mre = 0
# for i in annots['pred']:
#     mre = mre + np.abs(i[0][2] - i[1][2]) / i[1][2]
#
# mre / 25     ##mre

# mse = 0
# for i in annots['pred']:
#     mse = mse + np.square(i[0][2] - i[1][2])
#
# np.sqrt(mre / 25)

# mae = 0
# for i in annots['pred']:
#     mae = mae + np.abs(i[0][2] - i[1][2])
#
#mae / 25

# mre = 0
# xyz=0
# for j in range (13):
#     ab=0;
#     for i in annots['pred']:
#         if i[1][j]==0:
#             ab = ab + 0
#             xyz=xyz+1
#         else :
#             ab = ab + np.abs(i[0][j] - i[1][j]) / i[1][j]
#     mre = mre + ab
# mre = mre/(((25)*13)-xyz)

def get_points_from_desitymap(density_array):
    length=len(density_array)
    x_point=[]
    y_point=[]
    point_size=[]
    if (density_array[0][0]-density_array[0][1]>0.001 and density_array[0][0]-density_array[1][0]>0.001):
        x_point.append(0);
        y_point.append(0);
        point_size.append(density_array[0][1]+density_array[1][1]+density_array[1][0]+density_array[0][0]);
    if (density_array[0][length-1]-density_array[0][length-2]>0.001 and density_array[0][length-1]-density_array[1][length-1]>0.001):
        x_point.append(0);
        y_point.append(length-1);
        point_size.append(density_array[0][length-1] + density_array[0][length-2] + density_array[1][length-1]+density_array[1][length-2]);
    if (density_array[length-1][0]-density_array[length-2][0]>0.001 and density_array[length-1][0]-density_array[length-1][1]>0.001):
        x_point.append(length-1);
        y_point.append(0);
        point_size.append(density_array[length-1][0] + density_array[length-2][0] + density_array[length-1][1] +
                          density_array[length - 2][1]);
    if (density_array[length-1][length-1]-density_array[length-1][length-2]>0.001 and density_array[length-1][length-1]-density_array[length-2][length-1]>0.001):
        x_point.append(length-1);
        y_point.append(length-1);
        point_size.append(density_array[length-1][length-1] + density_array[length-1][length-2] + density_array[length-2][length-1] +
                          density_array[length - 2][length - 2]);
    for i in np.arange(1,length-1):  ## check four edges
        if (density_array[0][i] - density_array[0][i-1] > 0.001 and density_array[0][i] - density_array[0][i+1] > 0.001 and density_array[0][i] - density_array[1][i] > 0.001):
            x_point.append(0);
            y_point.append(i);
            point_size.append(density_array[0][i]+density_array[0][i-1]+density_array[0][i+1]+density_array[1][i-1]+density_array[1][i+1]+density_array[1][i])
        if (density_array[i][0] - density_array[i-1][0] > 0.001 and density_array[i][0] - density_array[i+1][0] > 0.001 and density_array[i][0] - density_array[i][1] > 0.001 ):
            x_point.append(i);
            y_point.append(0);
            point_size.append(density_array[i][0]+density_array[i-1][0]+density_array[i+1][0]+density_array[i][1]+density_array[i-1][1]+density_array[i+1][1])
        if (density_array[length - 1][i] - density_array[length - 1][i-1] > 0.001 and density_array[length - 1][i] -
                density_array[length - 1][i+1] > 0.001 and density_array[length - 1][i] - density_array[length - 2][i] > 0.001):
            x_point.append(length - 1);
            y_point.append(i);
            point_size.append(density_array[length - 1][i]+density_array[length - 1][i-1]+density_array[length - 1][i+1]+density_array[length - 2][i]+density_array[length - 2][i-1]+density_array[length - 2][i+1])
        if (density_array[i][length - 1] - density_array[i - 1][length - 1] > 0.001 and
                density_array[i][length - 1] - density_array[i + 1][length - 1] > 0.001 and density_array[i][length - 1] - density_array[i][length - 2] > 0.001):
            x_point.append(i);
            y_point.append(length - 1);
            point_size.append(density_array[i][length - 1]+density_array[i - 1][length - 1]+density_array[i +1][length - 1]+density_array[i][length - 2]+density_array[i - 1][length - 2]+density_array[i +1][length - 2])
    for i in np.arange(1,length-1):  ##check the middle part
        for j in np.arange(1,length-1):
            if (density_array[i][j]-density_array[i-1][j] > 0.001 and density_array[i][j]-density_array[i+1][j] > 0.001 and density_array[i][j]-density_array[i][j-1]> 0.001 and density_array[i][j]-density_array[i][j+1]> 0.001):
                x_point.append(i);
                y_point.append(j);
                point_size.append(density_array[i][j]+density_array[i-1][j]+density_array[i+1][j]+density_array[i-1][j-1]+density_array[i-1][j]+density_array[i-1][j+1]+density_array[i+1][j-1]+density_array[i+1][j]+density_array[i+1][j+1])
    df=pd.DataFrame(x_point, columns=['y'])
    df['x']=y_point
    df['Size']=point_size
    return df

def get_points_from_desitymap_21(density_array):  ## sum 21 boxes around the centre point
    length=len(density_array)
    x_point=[]
    y_point=[]
    point_size=[]
    if (density_array[0][0]-density_array[0][1]>0.001 and density_array[0][0]-density_array[1][0]>0.001):
        x_point.append(0);
        y_point.append(0);
        size_temp=density_array[0][1]+density_array[1][1]+density_array[1][0]+density_array[0][0]
        size_temp=size_temp+density_array[0][2]+size_temp+density_array[1][2]+density_array[2][0]+density_array[2][1]
        point_size.append(size_temp);
    if (density_array[0][length-1]-density_array[0][length-2]>0.001 and density_array[0][length-1]-density_array[1][length-1]>0.001):
        x_point.append(0);
        y_point.append(length-1);
        size_temp=density_array[0][length-1] + density_array[0][length-2] + density_array[1][length-1]+density_array[1][length-2]
        size_temp = size_temp + density_array[0][length-3]+ density_array[1][length-3]+density_array[2][length-1]+density_array[2][length-2]
        point_size.append(size_temp);
    if (density_array[length-1][0]-density_array[length-2][0]>0.001 and density_array[length-1][0]-density_array[length-1][1]>0.001):
        x_point.append(length-1);
        y_point.append(0);
        size_temp=density_array[length-1][0] + density_array[length-2][0] + density_array[length-1][1] +density_array[length - 2][1]
        size_temp = size_temp +density_array[length-3][0]+density_array[length-3][1]+density_array[length-1][2]+density_array[length - 2][2]
        point_size.append(size_temp);
    if (density_array[length-1][length-1]-density_array[length-1][length-2]>0.001 and density_array[length-1][length-1]-density_array[length-2][length-1]>0.001):
        x_point.append(length-1);
        y_point.append(length-1);
        size_temp=density_array[length-1][length-1] + density_array[length-1][length-2] + density_array[length-2][length-1] +density_array[length - 2][length - 2]
        size_temp =size_temp+ density_array[length-1][length-3]+density_array[length-2][length-3]+ density_array[length-3][length-1]+ density_array[length-3][length-2]
        point_size.append(size_temp);
    for i in np.arange(1,length-1):  ## check four edges
        if (density_array[0][i] - density_array[0][i-1] > 0.001 and density_array[0][i] - density_array[0][i+1] > 0.001 and density_array[0][i] - density_array[1][i] > 0.001):
            x_point.append(0);
            y_point.append(i);
            size_temp=density_array[0][i]+density_array[0][i-1]+density_array[0][i+1]+density_array[1][i-1]+density_array[1][i+1]+density_array[1][i]
            if i==1:
                size_temp=size_temp+density_array[2][i-1]+density_array[2][i]+density_array[2][i+1]+density_array[0][i+2]+density_array[1][i+2]
            elif i==length-2:
                size_temp =size_temp+density_array[2][length-1]+density_array[2][length-2]+density_array[2][length-3]+density_array[0][length-4]+density_array[1][length-4]
            else:
                size_temp = size_temp + size_temp+density_array[2][i-1]+density_array[2][i]+density_array[2][i+1]+density_array[0][i-2]+density_array[0][i+2]+density_array[1][i-2]+density_array[1][i+2]
            point_size.append(size_temp)
        if (density_array[i][0] - density_array[i-1][0] > 0.001 and density_array[i][0] - density_array[i+1][0] > 0.001 and density_array[i][0] - density_array[i][1] > 0.001 ):
            x_point.append(i);
            y_point.append(0);
            size_temp=density_array[i][0]+density_array[i-1][0]+density_array[i+1][0]+density_array[i][1]+density_array[i-1][1]+density_array[i+1][1]
            if i==1:
                size_temp=size_temp+density_array[3][0]+density_array[3][1]+density_array[0][2]+density_array[1][2]+density_array[2][2]
            elif i==length-2:
                size_temp = size_temp+density_array[length-4][0]+density_array[length-4][1]+density_array[length-1][2]+density_array[length-2][2]+density_array[length-3][2]
            else:
                size_temp = size_temp+density_array[i-2][0]+density_array[i-2][1]+density_array[i+2][0]+density_array[i+2][1]+density_array[i][2]+density_array[i-1][2]+density_array[i+1][2]
            point_size.append(size_temp)  ###continue...
        if (density_array[length - 1][i] - density_array[length - 1][i-1] > 0.001 and density_array[length - 1][i] -
                density_array[length - 1][i+1] > 0.001 and density_array[length - 1][i] - density_array[length - 2][i] > 0.001):
            x_point.append(length - 1);
            y_point.append(i);
            size_temp=density_array[length - 1][i]+density_array[length - 1][i-1]+density_array[length - 1][i+1]+density_array[length - 2][i]+density_array[length - 2][i-1]+density_array[length - 2][i+1]
            if i==1:
                size_temp=size_temp+density_array[length - 3][0]+density_array[length - 3][1]+density_array[length - 3][2]+density_array[length - 1][3]+density_array[length - 2][3]
            elif i==length-2:
                size_temp = size_temp+size_temp+density_array[length - 3][length - 3]+density_array[length - 3][length - 2]+density_array[length - 3][length - 1]+density_array[length - 1][length-4]+density_array[length - 2][length-4]
            point_size.append(size_temp)
        if (density_array[i][length - 1] - density_array[i - 1][length - 1] > 0.001 and
                density_array[i][length - 1] - density_array[i + 1][length - 1] > 0.001 and density_array[i][length - 1] - density_array[i][length - 2] > 0.001):
            x_point.append(i);
            y_point.append(length - 1);
            size_temp =density_array[i][length - 1]+density_array[i - 1][length - 1]+density_array[i +1][length - 1]+density_array[i][length - 2]+density_array[i - 1][length - 2]+density_array[i +1][length - 2]
            if i==1:
                size_temp=size_temp+density_array[3][length - 1]+density_array[3][length - 2]+density_array[0][length - 3]+density_array[1][length - 3]+density_array[2][length - 3]
            elif i==length-2:
                size_temp=size_temp+density_array[length-4][length - 1]+density_array[length-4][length - 2]+density_array[length-1][length - 3]+density_array[length-2][length - 3]+density_array[length-3][length - 3]
            else:
                size_temp = size_temp +density_array[i][length - 3]+density_array[i-1][length - 3]+density_array[i+1][length - 3]+density_array[i+2][length - 1]+density_array[i+2][length - 2]+density_array[i-2][length - 1]+density_array[i-2][length - 2]
            point_size.append(size_temp)
    for i in np.arange(1,length-1):  ##check the middle part
        for j in np.arange(1,length-1):
            if (density_array[i][j]-density_array[i-1][j] > 0.001 and density_array[i][j]-density_array[i+1][j] > 0.001 and density_array[i][j]-density_array[i][j-1]> 0.001 and density_array[i][j]-density_array[i][j+1]> 0.001):
                x_point.append(i);
                y_point.append(j);
                size_temp=density_array[i][j]+density_array[i-1][j]+density_array[i+1][j]+density_array[i-1][j-1]+density_array[i-1][j]+density_array[i-1][j+1]+density_array[i+1][j-1]+density_array[i+1][j]+density_array[i+1][j+1]
                if i-2>=0 and j-1>=0:
                    size_temp=size_temp+density_array[i-2][j-1]
                if i-2>=0:
                    size_temp=size_temp+density_array[i-2][j]
                if i-2>=0 and j+1<=length - 1:
                    size_temp=size_temp+density_array[i-2][j+1]
                if i-1>=0 and j+2<length - 1:
                    size_temp = size_temp + density_array[i - 1][j + 2]
                if j+2<length-1:
                    size_temp = size_temp + density_array[i][j + 2]
                if i+1<=length-1 and j+2<length - 1:
                    size_temp = size_temp + density_array[i+1][j + 2]
                if i+2<length - 1 and j+1<length - 1:
                    size_temp = size_temp + density_array[i + 2][j + 1]
                if i+2<length - 1:
                    size_temp = size_temp + density_array[i + 2][j]
                if i+2<length - 1 and j-1>=0:
                    size_temp = size_temp + density_array[i + 2][j-1]
                if i-1>=0 and j-2>=0:
                    size_temp = size_temp + density_array[i -1][j - 2]
                if j-2>=0:
                    size_temp = size_temp + density_array[i][j - 2]
                if j-2>0 and i+1<=length - 1:
                    size_temp = size_temp + density_array[i+1][j - 2]
                point_size.append(size_temp)
    df=pd.DataFrame(y_point, columns=['x'])
    df['y']=x_point
    df['Size']=point_size
    return df

#df.sum(axis = 0, skipna = True)
#df=df.sort_values(by=['x','y'])
#xss = df['x'].tolist()
#yss = df['y'].tolist()
#xss =[x*4 for x in xss]
#yss =[x*4 for x in yss]

#with open('C:\\Users\\Zixu.Liu\\AppData\\LocalLow\\DefaultCompany\\apple30\\20.txt') as f:
    #label = json.load(f)
#df_ori = pd.DataFrame (label,columns=['id','x','y','label','Size'])
#df_ori["Size"] = pd.to_numeric(df_ori["Size"], downcast="float")
#df_ori.sum(axis = 0, skipna = True)

def cal_size_error(df,df_ori):
        xss = df['x'].tolist()
        yss = df['y'].tolist()
        xss =[x*4 for x in xss]
        yss =[x*4 for x in yss]
        df['x_512'] = xss
        df['y_512'] = yss
        pre_index=[]
        dis=[]
        ori_index=[]
        if len(df_ori)>len(df):
            for index_df, point in df.iterrows():
                min_dis = 1000;
                nearst_index = 1000;
                for index_ori, item in df_ori.iterrows():
                    temp = np.sqrt(np.square(point['x_512'] - item['x']) + np.square(point['y_512'] - item['y']))
                    if temp < min_dis:
                        min_dis = temp
                        nearst_index = index_ori
                ori_index.append(nearst_index)
                pre_index.append(index_df)
                dis.append(min_dis)
            connection_df = pd.DataFrame(ori_index, columns=['ori_index'])
            connection_df['pre_index'] = pre_index
            connection_df['dis'] = dis
            for index_con, element in connection_df.iterrows():
                value = int(element['ori_index'])
                if value != 9999:
                    temp_df = connection_df.loc[connection_df['ori_index'] == value]
                    if len(temp_df) > 1:
                        closed_size = np.abs(df_ori.iloc[value]['Size'] - df.iloc[int(element['pre_index'])]['Size'])
                        closed_size_index = index_con
                        for t_index, t_ele in temp_df.iterrows():
                            temp_size_diff = np.abs(
                                df_ori.iloc[value]['Size'] - df.iloc[int(t_ele['pre_index'])]['Size'])
                            if closed_size > temp_size_diff:
                                closed_size = temp_size_diff;
                                closed_size_index = t_index
                        for t_index, t_ele in temp_df.iterrows():
                            if t_index != closed_size_index:
                                connection_df.at[t_index, 'ori_index'] = 9999;
            connection_df = connection_df.drop(connection_df[connection_df['ori_index'] == 9999].index)
        else:
            for index_ori, item in df_ori.iterrows():
                min_dis = 1000;
                nearst_index = 1000;
                for index_df, point in df.iterrows():
                    temp = np.sqrt(np.square(point['x_512'] - item['x']) + np.square(point['y_512'] - item['y']))
                    if temp < min_dis:
                        min_dis = temp
                        nearst_index = index_df
                ori_index.append(index_ori)
                pre_index.append(nearst_index)
                dis.append(min_dis)
            connection_df = pd.DataFrame(ori_index, columns=['ori_index'])
            connection_df['pre_index'] = pre_index
            connection_df['dis'] = dis
            for index_con, element in connection_df.iterrows():
                value = int(element['pre_index'])
                if value != 9999:
                    temp_df = connection_df.loc[connection_df['pre_index'] == value]
                    if len(temp_df) > 1:
                        closed_size = np.abs(
                                df.iloc[value]['Size'] - df_ori.iloc[int(element['ori_index'])]['Size'])
                        closed_size_index = index_con
                        for t_index, t_ele in temp_df.iterrows():
                            temp_size_diff = np.abs(
                                    df.iloc[value]['Size'] - df_ori.iloc[int(t_ele['ori_index'])]['Size'])
                            if closed_size > temp_size_diff:
                                closed_size = temp_size_diff;
                                closed_size_index = t_index
                        for t_index, t_ele in temp_df.iterrows():
                            if t_index != closed_size_index:
                                connection_df.at[t_index, 'pre_index'] = 9999;
            connection_df = connection_df.drop(connection_df[connection_df['pre_index'] == 9999].index)
        true_p=len(connection_df)
        false_p=len(df)-len(connection_df)
        false_n=len(df_ori)-len(connection_df)
        TPR=true_p/(true_p+false_n)   #recall
        PPV=true_p/(true_p+false_p)   # precision
        mre=0
        for index_df, point in connection_df.iterrows():
            t_ori_index=int(point['ori_index'])
            t_pre_index=int(point['pre_index'])
            mre = mre + np.abs(df.iloc[t_pre_index]['Size'] - df_ori.iloc[t_ori_index]['Size']) / df_ori.iloc[t_ori_index]['Size']

        mre = (mre ) / len(connection_df) ###
        return mre, true_p,false_n,false_p,connection_df,TPR,PPV

def get_all_mre(num_images,itemname,experiment):   #12
    path='C:\\Work\\qianproject\\\experiment_for_paper\\'+itemname+'\\'+experiment+'\\results_headCount_vgg16_bn_shA_size.mat'
    #annots = loadmat('C:\\Work\\qianproject\\\experiment_for_paper\\artichoke\\experiment 1\\results_headCount_vgg16_bn_shA_size.mat')
    annots = loadmat(path)
    #annots = loadmat('C:\\Work\\qianproject\\experiment size estimation\\results_headCount_vgg16_bn_shA.mat')
    mre_arr=[]
    true_p_a=[]
    false_n_a=[]
    false_p_a=[]
    TPR_a=[]  #TP/(TP+FN)
    PPV_a=[]  #TP/(TP+FP) accuracy
    f1_a=[]
    list = [13, 14, 18, 19, 23, 24, 28, 29, 3, 4, 8, 9]
    for i in range(num_images):
        df = get_points_from_desitymap_21(annots['densityArray'][i]) ## 15 sum
        #df = get_points_from_desitymap(annots['densityArray'][i]) ##  9 sum
        patho = 'C:\\Work\\qianproject\\experiment_for_paper\\'+itemname+'\\'
        #patho= 'C:\\Work\\qianproject\\experiment_for_paper\\artichoke\\'
        #path='C:\\Work\\qianproject\\experiment_for_paper\\apple\\'
        x=list[i]
        path=patho+'ground_truth\\'+str(x)+'.txt'
        print(path)
        with open(path) as f:
            label = json.load(f)
        df_ori = pd.DataFrame (label,columns=['id','x','y','label','Size'])
        df_ori["Size"] = pd.to_numeric(df_ori["Size"], downcast="float")
        mre, true_p,false_n,false_p,connection_df,TPR,PPV=cal_size_error(df,df_ori)
        mre_arr.append(mre)
        true_p_a.append(true_p)
        false_n_a.append(false_n)
        false_p_a.append(false_p)
        TPR_a.append(TPR)
        PPV_a.append(PPV)
        f1_a.append(2*TPR*PPV/(TPR+PPV))
    mre_df=pd.DataFrame(mre_arr,columns=['mre'])
    #mre_df['true_p']=true_p_a
    #mre_df['false_n'] = false_n_a
    #mre_df['false_p'] = false_p_a
    mre_df['recall'] = TPR_a
    mre_df['precision'] = PPV_a
    mre_df['F1']=f1_a
    ave_mre=mre_df["mre"].mean()
    ave_recall=mre_df["recall"].mean()
    ave_precision=mre_df['precision'].mean()
    ave_f1=mre_df['F1'].mean()
    mre_df.loc[mre_df.shape[0]] = [ave_mre, ave_recall, ave_precision,ave_f1]
    mre_df.to_csv(patho+experiment+'\\result.csv')
    return ave_mre, ave_recall, ave_precision,ave_f1

def output_to_paper(num_images,itemname):
    list_mre=[]
    list_recall=[]
    list_precision=[]
    list_f1=[]
    experiment='experiment '
    for i in range(3):
        t_exp=experiment+str(i+1)
        ave_mre, ave_recall, ave_precision,ave_f1=get_all_mre(num_images,itemname,t_exp)
        list_mre.append(ave_mre*100)
        list_recall.append(ave_recall*100)
        list_precision.append(ave_precision*100)
        list_f1.append(ave_f1*100)
    secon_half=" %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f " % (st.mean(list_precision),st.pstdev(list_precision),st.mean(list_recall),st.pstdev(list_recall),st.mean(list_f1),st.pstdev(list_f1),st.mean(list_mre),st.pstdev(list_mre))
    #print(secon_half)
    list_mae=[]
    list_rmse=[]
    list_mape=[]
    for i in range(3):
        t_exp = experiment + str(i + 1)
        path='C:\\Work\\qianproject\\\experiment_for_paper\\'+itemname+'\\'+t_exp+'\\results_headCount_vgg16_bn_shA_counting.mat'
        annots = loadmat(path)
        list_mae.append(annots['mae'][0][0])
        list_rmse.append(annots['mse'][0][0])
        list_mape.append(annots['mre'][0][0]*100)
    first_half=itemname.capitalize()+" & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f &" % (st.mean(list_mae),st.pstdev(list_mae),st.mean(list_rmse),st.pstdev(list_rmse),st.mean(list_mape),st.pstdev(list_mape))
    output_text=first_half+secon_half
    print(output_text)
    path_text='C:\\Work\\qianproject\\\experiment_for_paper\\'+itemname+'\\output.txt'
    f=open(path_text,'w')
    f.write(output_text)

def get_all_mre_size_exp(num_images,experiment_name,experiment):   #12
    path='C:\\Work\\qianproject\\carrot_backbone_test\\'+experiment_name+'\\'+experiment+'\\results_headCount_'+experiment_name+'_shA_size.mat'
    #annots = loadmat('C:\\Work\\qianproject\\\experiment_for_paper\\artichoke\\experiment 1\\results_headCount_vgg16_bn_shA_size.mat')
    annots = loadmat(path)
    #annots = loadmat('C:\\Work\\qianproject\\experiment size estimation\\results_headCount_vgg16_bn_shA.mat')
    mre_arr=[]
    true_p_a=[]
    false_n_a=[]
    false_p_a=[]
    TPR_a=[]  #TP/(TP+FN)
    PPV_a=[]  #TP/(TP+FP) accuracy
    f1_a=[]
    list = [13, 14, 18, 19, 23, 24, 28, 29, 3, 4, 8, 9]
    # if experiment_name=='small_medium':
    #     list=[10,11,12,13,14,25,26,27,28,29]
    # elif experiment_name=='small_large':
    #     list = [20,21,22,23,24,5,6,7,8,9]
    # else:   #medium_large
    #     list = [0,1,15,16,17,18,19,2,3,4]
    # if experiment_name=='6':
    #     list=[1,11,16,21,26,6]
    # elif experiment_name=='60':
    #     list = [x+60 for x in range(30)]
    # elif experiment_name=='30':   #medium_large
    #     list = [30,31,32,35,36,37,40,41,42,45,46,47,50,51,52,55,56,57]
    for i in range(num_images):
        df = get_points_from_desitymap_21(annots['densityArray'][i]) ## 15 sum   ## annots['densityArray'][i][0] for inception
        print(annots['densityArray'][i])
        #df = get_points_from_desitymap(annots['densityArray'][i]) ##  9 sum
        patho = 'C:\\Work\\qianproject\\carrot_backbone_test\\'+experiment_name+'\\'
        #patho= 'C:\\Work\\qianproject\\experiment_for_paper\\artichoke\\'
        #path='C:\\Work\\qianproject\\experiment_for_paper\\apple\\'
        x=list[i]
        path=patho+'ground_truth\\'+str(x)+'.txt'
        print(path)
        with open(path) as f:
            label = json.load(f)
        df_ori = pd.DataFrame (label,columns=['id','x','y','label','Size'])
        df_ori["Size"] = pd.to_numeric(df_ori["Size"], downcast="float")
        print(len(df_ori))
        print(len(df))
        mre, true_p,false_n,false_p,connection_df,TPR,PPV=cal_size_error(df,df_ori)
        mre_arr.append(mre)
        true_p_a.append(true_p)
        false_n_a.append(false_n)
        false_p_a.append(false_p)
        TPR_a.append(TPR)
        PPV_a.append(PPV)
        f1_a.append(2*TPR*PPV/(TPR+PPV))
    mre_df=pd.DataFrame(mre_arr,columns=['mre'])
    #mre_df['true_p']=true_p_a
    #mre_df['false_n'] = false_n_a
    #mre_df['false_p'] = false_p_a
    mre_df['recall'] = TPR_a
    mre_df['precision'] = PPV_a
    mre_df['F1']=f1_a
    ave_mre=mre_df["mre"].mean()
    ave_recall=mre_df["recall"].mean()
    ave_precision=mre_df['precision'].mean()
    ave_f1=mre_df['F1'].mean()
    mre_df.loc[mre_df.shape[0]] = [ave_mre, ave_recall, ave_precision,ave_f1]
    mre_df.to_csv(patho+experiment+'\\result.csv')
    return ave_mre, ave_recall, ave_precision,ave_f1

def output_to_paper_size_exp(num_images,experiment_name):
    list_mre=[]
    list_recall=[]
    list_precision=[]
    list_f1=[]
    experiment='experiment '
    for i in range(3):
        t_exp=experiment+str(i+1)
        ave_mre, ave_recall, ave_precision,ave_f1=get_all_mre_size_exp(num_images,experiment_name,t_exp)
        list_mre.append(ave_mre*100)
        list_recall.append(ave_recall*100)
        list_precision.append(ave_precision*100)
        list_f1.append(ave_f1*100)
    secon_half=" %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f " % (st.mean(list_precision),st.pstdev(list_precision),st.mean(list_recall),st.pstdev(list_recall),st.mean(list_f1),st.pstdev(list_f1),st.mean(list_mre),st.pstdev(list_mre))
    #print(secon_half)
    list_mae=[]
    list_rmse=[]
    list_mape=[]
    for i in range(3):
        t_exp = experiment + str(i + 1)
        path='C:\\Work\\qianproject\\carrot_backbone_test\\'+experiment_name+'\\'+t_exp+'\\results_headCount_'+experiment_name+'_shA_counting.mat'
        annots = loadmat(path)
        list_mae.append(annots['mae'][0][0])
        list_rmse.append(annots['mse'][0][0])
        list_mape.append(annots['mre'][0][0]*100)
    #experiment_name.capitalize()
    first_half=experiment_name+" & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f &" % (st.mean(list_mae),st.pstdev(list_mae),st.mean(list_rmse),st.pstdev(list_rmse),st.mean(list_mape),st.pstdev(list_mape))
    output_text=first_half+secon_half
    print(output_text)
    path_text='C:\\Work\\qianproject\\carrot_backbone_test\\'+experiment_name+'\\output.txt'
    f=open(path_text,'w')
    f.write(output_text)

def get_all_mre_multi(num_images,num_items,experiment_name,experiment,item_names):   #12
    path='C:\\Work\\qianproject\\multi_class_experiment\\'+experiment_name+'\\'+experiment+'\\results_headCount_vgg16_bn_size.mat'
    #annots = loadmat('C:\\Work\\qianproject\\\experiment_for_paper\\artichoke\\experiment 1\\results_headCount_vgg16_bn_shA_size.mat')
    annots = loadmat(path)
    #annots = loadmat('C:\\Work\\qianproject\\experiment size estimation\\results_headCount_vgg16_bn_shA.mat')
    mre_arr=[]
    true_p_a=[]
    false_n_a=[]
    false_p_a=[]
    TPR_a=[]  #TP/(TP+FN)
    PPV_a=[]  #TP/(TP+FP) accuracy
    f1_a=[]
    list = [13, 14, 18, 19, 23, 24, 28, 29, 3, 4, 8, 9]
    # if experiment_name=='small_medium':
    #     list=[10,11,12,13,14,25,26,27,28,29]
    # elif experiment_name=='small_large':
    #     list = [20,21,22,23,24,5,6,7,8,9]
    # else:   #medium_large
    #     list = [0,1,15,16,17,18,19,2,3,4]
    # if experiment_name=='6':
    #     list=[1,11,16,21,26,6]
    # elif experiment_name=='60':
    #     list = [x+60 for x in range(30)]
    # elif experiment_name=='30':   #medium_large
    #     list = [30,31,32,35,36,37,40,41,42,45,46,47,50,51,52,55,56,57]
    for i in range(num_images):
        mre_o=0
        true_p_o=0
        false_n_o=0
        false_p_o=0
        for j in range (num_items):
            df = get_points_from_desitymap_21(annots['densityArray'][i][j]) ## 15 sum   ## annots['densityArray'][i][0] for inception
            #print(annots['densityArray'][i])
            #df = get_points_from_desitymap(annots['densityArray'][i]) ##  9 sum
            patho = 'C:\\Work\\qianproject\\multi_class_experiment\\'+experiment_name+'\\'
        #patho= 'C:\\Work\\qianproject\\experiment_for_paper\\artichoke\\'
        #path='C:\\Work\\qianproject\\experiment_for_paper\\apple\\'
            x=list[i]
            path=patho+'ground_truth\\'+str(x)+'.txt'
            with open(path) as f:
                label = json.load(f)
            label=[item for item in label if item_names[j] in item['label']]
            df_ori = pd.DataFrame (label,columns=['id','x','y','label','Size'])
            df_ori["Size"] = pd.to_numeric(df_ori["Size"], downcast="float")
            mre, true_p,false_n,false_p,connection_df,TPR,PPV=cal_size_error(df,df_ori)
            mre_o=mre_o+mre
            true_p_o=true_p+true_p_o
            false_n_o=false_n_o+false_n
            false_p_o=false_p_o+false_p
        TPR = true_p_o / (true_p_o + false_n_o)  # recall
        PPV = true_p_o / (true_p_o + false_p_o)
        mre_arr.append(mre_o/num_items)
        true_p_a.append(true_p_o)
        false_n_a.append(false_n_o)
        false_p_a.append(false_p_o)
        TPR_a.append(TPR)
        PPV_a.append(PPV)
        f1_a.append(2*TPR*PPV/(TPR+PPV))
    mre_df=pd.DataFrame(mre_arr,columns=['mre'])
    #mre_df['true_p']=true_p_a
    #mre_df['false_n'] = false_n_a
    #mre_df['false_p'] = false_p_a
    mre_df['recall'] = TPR_a
    mre_df['precision'] = PPV_a
    mre_df['F1']=f1_a
    ave_mre=mre_df["mre"].mean()
    ave_recall=mre_df["recall"].mean()
    ave_precision=mre_df['precision'].mean()
    ave_f1=mre_df['F1'].mean()
    mre_df.loc[mre_df.shape[0]] = [ave_mre, ave_recall, ave_precision,ave_f1]
    mre_df.to_csv(patho+experiment+'\\result.csv')
    return ave_mre, ave_recall, ave_precision,ave_f1

def output_to_paper_multi(num_images,num_items,experiment_name,item_names=['_banana','_avocado','_artichoke']):
    list_mre=[]
    list_recall=[]
    list_precision=[]
    list_f1=[]
    experiment='experiment '
    for i in range(3):
        t_exp=experiment+str(i+1)
        ave_mre, ave_recall, ave_precision,ave_f1=get_all_mre_multi(num_images,num_items,experiment_name,t_exp,item_names)
        list_mre.append(ave_mre*100)
        list_recall.append(ave_recall*100)
        list_precision.append(ave_precision*100)
        list_f1.append(ave_f1*100)
    secon_half=" %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f " % (st.mean(list_precision),st.pstdev(list_precision),st.mean(list_recall),st.pstdev(list_recall),st.mean(list_f1),st.pstdev(list_f1),st.mean(list_mre),st.pstdev(list_mre))
    #print(secon_half)
    list_mae=[]
    list_rmse=[]
    list_mape=[]
    for i in range(3):
        t_exp = experiment + str(i + 1)
        path='C:\\Work\\qianproject\\multi_class_experiment\\'+experiment_name+'\\'+t_exp+'\\results_headCount_vgg16_bn_counting.mat'
        annots = loadmat(path)
        list_mae.append(annots['mae'][0][0])
        list_rmse.append(np.sqrt(np.square(annots['mse'][0][0])*num_items)/num_items)
        list_mape.append(annots['mre'][0][0]*100)
    #experiment_name.capitalize()
    first_half=experiment_name+" & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f &" % (st.mean(list_mae),st.pstdev(list_mae),st.mean(list_rmse),st.pstdev(list_rmse),st.mean(list_mape),st.pstdev(list_mape))
    output_text=first_half+secon_half
    print(output_text)
    path_text='C:\\Work\\qianproject\\multi_class_experiment\\'+experiment_name+'\\output.txt'
    f=open(path_text,'w')
    f.write(output_text)

def get_all_mre_num_exp(num_images,experiment_name,experiment):   #12
    path='C:\\Work\\qianproject\\Banana_number_test\\'+experiment_name+'\\'+experiment+'\\results_headCount_vgg16_bn_shA_size.mat'
    #annots = loadmat('C:\\Work\\qianproject\\\experiment_for_paper\\artichoke\\experiment 1\\results_headCount_vgg16_bn_shA_size.mat')
    annots = loadmat(path)
    #annots = loadmat('C:\\Work\\qianproject\\experiment size estimation\\results_headCount_vgg16_bn_shA.mat')
    mre_arr=[]
    true_p_a=[]
    false_n_a=[]
    false_p_a=[]
    TPR_a=[]  #TP/(TP+FN)
    PPV_a=[]  #TP/(TP+FP) accuracy
    f1_a=[]
    list = [32,33,34,37,38,39,42,43,44,47,48,49,52,53,54,57,58,59,60,61,65,66,70,71,75,76,80,81,85,86]
    # if experiment_name=='small_medium':
    #     list=[10,11,12,13,14,25,26,27,28,29]
    # elif experiment_name=='small_large':
    #     list = [20,21,22,23,24,5,6,7,8,9]
    # else:   #medium_large
    #     list = [0,1,15,16,17,18,19,2,3,4]
    # if experiment_name=='6':
    #     list=[1,11,16,21,26,6]
    # elif experiment_name=='60':
    #     list = [x+60 for x in range(30)]
    # elif experiment_name=='30':   #medium_large
    #     list = [30,31,32,35,36,37,40,41,42,45,46,47,50,51,52,55,56,57]
    for i in range(num_images):
        df = get_points_from_desitymap_21(annots['densityArray'][i]) ## 15 sum   ## annots['densityArray'][i][0] for inception
        #df = get_points_from_desitymap(annots['densityArray'][i]) ##  9 sum
        patho = 'C:\\Work\\qianproject\\Banana_number_test\\'+experiment_name+'\\'
        #patho= 'C:\\Work\\qianproject\\experiment_for_paper\\artichoke\\'
        #path='C:\\Work\\qianproject\\experiment_for_paper\\apple\\'
        x=list[i]
        path=patho+'ground_truth\\'+str(x)+'.txt'
        print(path)
        with open(path) as f:
            label = json.load(f)
        df_ori = pd.DataFrame (label,columns=['id','x','y','label','Size'])
        df_ori["Size"] = pd.to_numeric(df_ori["Size"], downcast="float")
        mre, true_p,false_n,false_p,connection_df,TPR,PPV=cal_size_error(df,df_ori)
        mre_arr.append(mre)
        true_p_a.append(true_p)
        false_n_a.append(false_n)
        false_p_a.append(false_p)
        TPR_a.append(TPR)
        PPV_a.append(PPV)
        f1_a.append(2*TPR*PPV/(TPR+PPV))
    mre_df=pd.DataFrame(mre_arr,columns=['mre'])
    #mre_df['true_p']=true_p_a
    #mre_df['false_n'] = false_n_a
    #mre_df['false_p'] = false_p_a
    mre_df['recall'] = TPR_a
    mre_df['precision'] = PPV_a
    mre_df['F1']=f1_a
    ave_mre=mre_df["mre"].mean()
    ave_recall=mre_df["recall"].mean()
    ave_precision=mre_df['precision'].mean()
    ave_f1=mre_df['F1'].mean()
    mre_df.loc[mre_df.shape[0]] = [ave_mre, ave_recall, ave_precision,ave_f1]
    mre_df.to_csv(patho+experiment+'\\result.csv')
    return ave_mre, ave_recall, ave_precision,ave_f1

def output_to_paper_num_exp(num_images,experiment_name):
    list_mre=[]
    list_recall=[]
    list_precision=[]
    list_f1=[]
    experiment='experiment '
    for i in range(3):
        t_exp=experiment+str(i+1)
        ave_mre, ave_recall, ave_precision,ave_f1=get_all_mre_num_exp(num_images,experiment_name,t_exp)
        list_mre.append(ave_mre*100)
        list_recall.append(ave_recall*100)
        list_precision.append(ave_precision*100)
        list_f1.append(ave_f1*100)
    secon_half=" %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f " % (st.mean(list_precision),st.pstdev(list_precision),st.mean(list_recall),st.pstdev(list_recall),st.mean(list_f1),st.pstdev(list_f1),st.mean(list_mre),st.pstdev(list_mre))
    #print(secon_half)
    list_mae=[]
    list_rmse=[]
    list_mape=[]
    for i in range(3):
        t_exp = experiment + str(i + 1)
        path='C:\\Work\\qianproject\\Banana_number_test\\'+experiment_name+'\\'+t_exp+'\\results_headCount_vgg16_bn_shA_counting.mat'
        annots = loadmat(path)
        list_mae.append(annots['mae'][0][0])
        list_rmse.append(annots['mse'][0][0])
        list_mape.append(annots['mre'][0][0]*100)
    #experiment_name.capitalize()
    first_half=experiment_name+" & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f &" % (st.mean(list_mae),st.pstdev(list_mae),st.mean(list_rmse),st.pstdev(list_rmse),st.mean(list_mape),st.pstdev(list_mape))
    output_text=first_half+secon_half
    print(output_text)
    path_text='C:\\Work\\qianproject\\Banana_number_test\\'+experiment_name+'\\output.txt'
    f=open(path_text,'w')
    f.write(output_text)

def output_to_paper_counting_exp(experiment_name):
    experiment='experiment '
    secon_half=" N.A & N.A & N.A & N.A "
    #print(secon_half)
    list_mae=[]
    list_rmse=[]
    list_mape=[]
    for i in range(3):
        t_exp = experiment + str(i + 1)
        path='C:\\Work\\qianproject\\carrot_backbone_test\\'+experiment_name+'\\'+t_exp+'\\results_headCount_'+experiment_name+'_shA_counting.mat'
        annots = loadmat(path)
        list_mae.append(annots['mae'][0][0])
        list_rmse.append(annots['mse'][0][0])
        list_mape.append(annots['mre'][0][0]*100)
    #experiment_name.capitalize()
    first_half=experiment_name+" & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f &" % (st.mean(list_mae),st.pstdev(list_mae),st.mean(list_rmse),st.pstdev(list_rmse),st.mean(list_mape),st.pstdev(list_mape))
    output_text=first_half+secon_half
    print(output_text)
    path_text='C:\\Work\\qianproject\\carrot_backbone_test\\'+experiment_name+'\\output.txt'
    f=open(path_text,'w')
    f.write(output_text)
