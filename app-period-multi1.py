#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
import inventorize as inv
from PIL import Image
import math


# In[2]:


image1 = Image.open('logigeek-logo-short.png')
st.set_page_config(
    page_title="app-period-multi|LogiGeek", 
    page_icon=image1,
    layout="wide")

image2 = Image.open('logigeek-logo-long.png')
st.image(image2, caption='ロジスティクスをDXするための小ネタ集')
st.link_button(':blue[:link:ロジギークのHPへ]', 
               'https://rikei-logistics.com/',
                use_container_width = True)

st.header(':blue[適正発注を行った場合のトータル物流コストを計算するアプリ]')
st.text('')
st.subheader('このアプリでできること', divider='rainbow')
st.text('１．安全在庫理論に基づく定期発注を行った場合のトータル調達物流コストを算出します。')
st.text('２．調達先からの輸送コストと倉庫への入庫コスト、倉庫での保管コスト、欠品による機会損失コストを考慮しています。')
st.text('３．発注先別にリードタイム、発注サイクル、許容欠品率を設定できる他、発注日が重ならないようにズラすこともできます。')
st.text('４．商品アイテム別の需要データをcsvファイルでアップロードすることができます。')
st.text('５．商品アイテム別に容積(m3)、重量(kg)、価格(円)、欠品コスト(円)、発注先をcsvファイルでアップロードすることができます。')
st.text('６．トラック車種別／発注先別の車建て輸送料金をcsvファイルでアップロードすることができます。')
st.text('７．倉庫への入庫単価(円/m3)と保管単価(円/m3･日)を設定することができます。')
st.text('８．物流コストは発注先別の他、日別／商品アイテム別にも閲覧／データダウンロードできます。')
st.text('９．全商品アイテムの在庫推移をグラフで閲覧することができます。')
st.text('詳細な使い方については下記サイトをご覧下さい↓')
st.link_button(":blue[:link:【これは便利！】適正発注を行った場合のトータル調達物流コストを計算するアプリ|ロジギーク]", 
               "https://rikei-logistics.com/app-multi-sku1")

# データの読み込みとパラメータの設定
st.text('')
st.sidebar.header('◆条件設定画面◆')
st.sidebar.subheader('１．需要データの読み込み')
uploaded_file = st.sidebar.file_uploader('需要データをアップロードして下さい。',type='csv')

if uploaded_file:
    raw_df0 = pd.read_csv(uploaded_file)
else:
    raw_df0 = pd.read_csv('default_data4.csv')
    
st.sidebar.subheader('２．マスタデータの読み込み')
st.sidebar.markdown('''#### １）商品マスタ''')
uploaded_file2 = st.sidebar.file_uploader('商品マスタをアップロードして下さい。',type='csv')

if uploaded_file2:
    master_df0 = pd.read_csv(uploaded_file2)
else:
    master_df0 = pd.read_csv('master_data4.csv')
    
st.sidebar.markdown('''#### ２）輸送マスタ''')
uploaded_file3 = st.sidebar.file_uploader('輸送マスタをアップロードして下さい。',type='csv')

if uploaded_file3:
    fleet_df = pd.read_csv(uploaded_file3)
else:
    fleet_df = pd.read_csv('fleet_data.csv')

if len(fleet_df) > 5:
    st.sidebar.error('トラックは５車種以内で設定して下さい。')
    
st.sidebar.subheader('３．訓練データの割合')
train_ratio = st.sidebar.number_input(label = '訓練データ（％）', 
                                     value = 40, label_visibility="visible", 
                                     min_value=0, max_value=100)

st.sidebar.subheader('４．発注関連パラメータ')
v_number = len(master_df0.Vendor.unique())
st.sidebar.write('発注先は', f':orange[{v_number}社]です。')

ld_list = []
oc_list = []
so_list = []
off_set_list = []
for vendor in master_df0.Vendor.unique():
    ld = st.sidebar.number_input(label = '納品リードタイム（日）:point_right: ' f'{vendor}', 
                                         value = 2, label_visibility="visible", 
                                         min_value=0, max_value=180)
    oc = st.sidebar.number_input(label = '発注サイクル（日）:point_right: ' f'{vendor}', 
                                         value = 3, label_visibility="visible", 
                                         min_value=0, max_value=180)
    so = st.sidebar.number_input(label = '許容欠品率（％）:point_right: ' f'{vendor}', 
                                         value = 3, label_visibility="visible", 
                                         min_value=0, max_value=100)
    off_set = st.sidebar.number_input(label = '発注日オフセット（日）:point_right: ' f'{vendor}', 
                                         value = 0, label_visibility="visible", 
                                         min_value=0, max_value=100)
    st.sidebar.text('-----------------------')
    
    ld_list.append(ld)
    oc_list.append(oc)
    so_list.append(so)
    off_set_list.append(off_set)
    

st.sidebar.subheader('５．倉庫関連コスト')
st.sidebar.markdown('''#### １）入庫コスト''')
handling_cost = st.sidebar.number_input(label = '単価（円／M3）', 
                                     value = 600, label_visibility="visible", 
                                     min_value=0, max_value=10000)
st.sidebar.markdown('''#### ２）保管コスト''')
storage_cost = st.sidebar.number_input(label = '単価（円／M3･日）', 
                                     value = 80, label_visibility="visible", 
                                     min_value=0, max_value=1000)

#　読み込みデータの表示
st.subheader('読み込みデータ一覧', divider='rainbow')
st.subheader(':mag:需要データ')
st.write('アイテム数　：', f':orange[{raw_df0.shape[1]}個]')
st.write('日数　：', f':orange[{raw_df0.shape[0]}日分]')
index_list = []
for i in range(len(raw_df0)):
    index_list.append(f'{i+1}日目')
raw_df0.index = index_list
st.dataframe(raw_df0, hide_index = False)

col1, col2 = st.columns((1, 1))
with col1:
    st.subheader(':mag:商品マスタ')
    st.write('アイテム数　：', f':orange[{len(master_df0)}個]')
    st.dataframe(master_df0, hide_index = True)

with col2:
    st.subheader(':mag:輸送マスタ')
    st.write('トラック車種　：', f':orange[{fleet_df.shape[1]}種類]')
    st.write('発注先数　：', f':orange[{fleet_df.shape[0]-2}社]')
    st.dataframe(fleet_df, hide_index = True)

#　シミュレーション
vendor_dict = master_df0.set_index('SKU').drop(['UnitM3', 'UnitKG', 'UnitPrice', 'UnitGP'], axis = 1).to_dict()
a = pd.DataFrame(raw_df0.columns.map(vendor_dict['Vendor'])).T
a.columns = raw_df0.columns
b = pd.concat([raw_df0 , a]).reset_index(drop = True)
r = b.shape[0]

st.subheader('シミュレーション結果', divider='rainbow')

tab1, tab2 = st.tabs(['全体', '明細'])

#　計算根拠を表示
with tab2:
    n = 0
    result_df = pd.DataFrame()
    store_m3_all_df = pd.DataFrame()
    for vendor in master_df0.Vendor.unique():
        raw_df0 = b.loc[:, b.loc[r-1,:] == vendor].iloc[:-1, :]
        ld = ld_list[n]
        oc = oc_list[n]
        so = so_list[n]
        off_set = off_set_list[n]
        n = n + 1

        order_m3_df = pd.DataFrame()
        store_m3_df = pd.DataFrame()
        stock_out_df = pd.DataFrame()
        stock_out_df2 = pd.DataFrame()
        for sku in range(len(raw_df0.columns)):
            raw_df = raw_df0.iloc[:, sku]
            raw_df = np.array(raw_df)

            learn_df, test_df = train_test_split(raw_df, train_size=train_ratio/100, shuffle=False)

            av = learn_df.mean()
            sd = learn_df.std(ddof = 1)

            result = inv.Periodic_review_normal(
                test_df,
                av,
                sd,
                ld,
                1-so/100,
                oc,
                shortage_cost = 0,
                inventory_cost = 0,
                ordering_cost = 0
            )

            unit_m3 = master_df0[master_df0['SKU'] == raw_df0.columns[sku]].UnitM3
            gp = master_df0[master_df0['SKU'] == raw_df0.columns[sku]].UnitGP
            vendor_name = master_df0[master_df0['SKU'] == raw_df0.columns[sku]].Vendor
            show_df = result[0].rename(columns={'period': '日', 'demand': '需要', 'sales': '出荷', 'inventory_level': '庫内在庫',
                                               'inventory_position': 'トータル在庫', 'order': '発注', 'max': '補充目標', 'recieved': '入庫',
                                                'lost_order': '欠品'})
            for i in range(off_set):
                x = np.zeros(len(show_df.columns)).reshape(1,len(show_df.columns))
                df_insert = pd.DataFrame(x, columns=show_df.columns)
                show_df = pd.concat([df_insert, show_df])
                show_df = show_df[:-1]
            show_df.reset_index(inplace = True, drop = True)

            show_df['OrderM3'] = 0
            for i in range(show_df.shape[0]):
                if show_df.iloc[i, 5] != 0:
                    show_df.iloc[i, 9] = unit_m3 * show_df.iloc[i, 5]
            order_m3_list = show_df.OrderM3

            show_df['StoreM3'] = 0
            for i in range(show_df.shape[0]):
                show_df.iloc[i, 10] = unit_m3 * show_df.iloc[i, 3]
            store_m3_list = show_df.StoreM3

            show_df['StockOut'] = 0
            for i in range(show_df.shape[0]):
                show_df.iloc[i, 11] = show_df.iloc[i, 8] * gp
            stock_out_list = show_df.StockOut

            order_m3_df = order_m3_df.append(order_m3_list)
            store_m3_df = store_m3_df.append(store_m3_list)
            stock_out_df = stock_out_df.append(stock_out_list)
            stock_out_df2 = stock_out_df2.append(show_df.欠品)

        order_m3_df = order_m3_df.append(pd.DataFrame(np.sum(order_m3_df)).T).reset_index(drop = True)
        sku_list = np.array(raw_df0.columns).tolist()
        sku_list += ['Total']
        order_m3_df['SKU'] = sku_list

        st.subheader(f':mag: {vendor}社調達品にかかる物流コストの計算根拠')

        def step(x):
            return np.where(x > 0, 1, 0)

        a = order_m3_df.shape[0]
        bar = np.array(fleet_df['M3']).tolist()
        order_m3_df = order_m3_df.reindex(index=range(len(bar)+a+3))
        order_m3_df.iloc[a, :-1] = order_m3_df.iloc[a-1, :-1] // bar[0]
        order_m3_df.iloc[a+1, :-1] = (order_m3_df.iloc[a-1, :-1] % bar[0]) // bar[1]
        order_m3_df.iloc[a+2, :-1] = ((order_m3_df.iloc[a-1, :-1] % bar[0]) % bar[1]) // bar[2]
        order_m3_df.iloc[a+3, :-1] = (((order_m3_df.iloc[a-1, :-1] % bar[0]) % bar[1]) % bar[2]) // bar[3]
        order_m3_df.iloc[a+4, :-1] = (((order_m3_df.iloc[a-1, :-1] % bar[0]) % bar[1]) % bar[2]) % bar[3] // bar[4]
        order_m3_df.iloc[a+5, :-1] = step((((order_m3_df.iloc[a-1, :-1] % bar[0]) % bar[1]) % bar[2]) % bar[3] % bar[4])
        order_m3_df.iloc[a+6, :-1] = fleet_df.iloc[0, n+1] * order_m3_df.iloc[a, :-1] + fleet_df.iloc[1, n+1] * order_m3_df.iloc[a+1, :-1] + fleet_df.iloc[2, n+1] * order_m3_df.iloc[a+2, :-1] + fleet_df.iloc[3, n+1] * order_m3_df.iloc[a+3, :-1] + fleet_df.iloc[4, n+1] * order_m3_df.iloc[a+4, :-1] + fleet_df.iloc[4, n+1] * order_m3_df.iloc[a+5, :-1]
        order_m3_df.iloc[a+7, :-1] = handling_cost * order_m3_df.iloc[a, :-1]
        order_m3_df.iloc[a+0, -1] = f'{fleet_df.iloc[0,0]}トラックの台数'
        order_m3_df.iloc[a+1, -1] = f'{fleet_df.iloc[1,0]}トラックの台数'
        order_m3_df.iloc[a+2, -1] = f'{fleet_df.iloc[2,0]}トラックの台数'
        order_m3_df.iloc[a+3, -1] = f'{fleet_df.iloc[3,0]}トラックの台数'
        order_m3_df.iloc[a+4, -1] = f'{fleet_df.iloc[4,0]}トラックの台数'
        order_m3_df.iloc[a+5, -1] = f'{fleet_df.iloc[4,0]}トラックの追加台数'
        order_m3_df.iloc[a+6, -1] = f'1){vendor}社調達品の輸送コスト'
        order_m3_df.iloc[a+7, -1] = f'2){vendor}社調達品の入庫コスト'

        order_m3_df.insert(loc=0, column='SKU2', value = 0)
        order_m3_df.SKU2 = order_m3_df.SKU
        order_m3_df.pop('SKU')
        order_m3_df = order_m3_df.rename(columns={'SKU2': 'SKU'})

        col_list = ['SKU']
        for i in range(len(test_df)+1):
            col_list.append(f'{i}日目')
        order_m3_df.columns = col_list

        st.markdown(':point_right:***発注M3と輸送 ＆ 入庫コスト***')
        st.dataframe(order_m3_df, hide_index = True)

        store_m3_df = store_m3_df.append(pd.DataFrame(np.sum(store_m3_df)).T).reset_index(drop = True)
        store_m3_df['SKU'] = sku_list

        bb = store_m3_df.shape[0]
        store_m3_df = store_m3_df.reindex(index=range(bb+1))
        store_m3_df.iloc[bb, :-1] = store_m3_df.iloc[bb-1, :-1] * storage_cost
        store_m3_df.iloc[bb, -1] = f'3){vendor}社調達品の保管コスト'

        store_m3_df.insert(loc=0, column='SKU2', value = 0)
        store_m3_df.SKU2 = store_m3_df.SKU
        store_m3_df.pop('SKU')
        store_m3_df = store_m3_df.rename(columns={'SKU2': 'SKU'})

        temp_df = store_m3_df[:-1][:-1]
        store_m3_all_df = pd.concat([store_m3_all_df, temp_df]).reset_index(drop = True)
        store_m3_df.columns = col_list

        st.markdown(':point_right:***在庫M3と保管コスト***')
        st.dataframe(store_m3_df, hide_index = True)

        stock_out_df = stock_out_df.append(pd.DataFrame(np.sum(stock_out_df)).T).reset_index(drop = True)
        stock_out_df2 = stock_out_df2.append(pd.DataFrame(np.sum(stock_out_df2)).T).reset_index(drop = True)
        temp1_df = stock_out_df2.iloc[:-1, :]
        temp2 = stock_out_df.iloc[-1, :]
        temp2_df = pd.DataFrame(temp2).T
        stock_out_df = pd.concat([temp1_df, temp2_df]).reset_index(drop = True)
        stock_out_df['SKU'] = sku_list

        c = stock_out_df.shape[0]
        stock_out_df = stock_out_df.reindex(index=range(c+2))
        stock_out_df.iloc[c, :-1] = stock_out_df.iloc[c-1, :-1]
        stock_out_df.iloc[c-1, :-1] = stock_out_df2.iloc[-1, :]
        stock_out_df.iloc[c, -1] = f'4){vendor}社調達品の欠品コスト'
        stock_out_df.iloc[c+1, -1] = f'5){vendor}社調達品の総物流コスト'

        stock_out_df.insert(loc=0, column='SKU2', value = 0)
        stock_out_df.SKU2 = stock_out_df.SKU
        stock_out_df.pop('SKU')
        stock_out_df = stock_out_df.rename(columns={'SKU2': 'SKU'})

        stock_out_df.columns = col_list

        st.markdown(':point_right:***欠品数量と欠品コスト***')
        st.dataframe(stock_out_df, hide_index = True)

        st.text('---------------------------------------------------------------------------')

        result_df = pd.concat([result_df, order_m3_df.iloc[a+6:, :]])
        result_df = pd.concat([result_df, store_m3_df.iloc[bb:, :]])
        result_df = pd.concat([result_df, stock_out_df.iloc[c:, :]])

#　集計結果を表示
with tab1:
    result_df.reset_index(inplace = True, drop = True)
    for v in range(v_number):
        result_df.iloc[4 + v * 5, 1:] = result_df.iloc[4 + v * 5 - 4, 1:] + result_df.iloc[4 + v * 5 - 3, 1:] + result_df.iloc[4 + v * 5 - 2, 1:] + result_df.iloc[4 + v * 5 - 1, 1:]

    result_df = result_df.assign(合計 = result_df.iloc[:, :-1].sum(axis=1))

    row = result_df.shape[0]
    col = result_df.shape[1]
    result_df = result_df.reindex(index=range(row + 5))
    result_df.iloc[row, 0] = '1)輸送コスト合計'
    result_df.iloc[row + 1, 0] = '2)入庫コスト合計'
    result_df.iloc[row + 2, 0] = '3)保管コスト合計'
    result_df.iloc[row + 3, 0] = '4)欠品コスト合計'
    result_df.iloc[row + 4, 0] = '5)物流コスト合計'
    x1 = 0
    for v in range(v_number):
        x1 = x1 + result_df.iloc[0 + 5 * v, col - 1]
    result_df.iloc[row, col - 1] = x1
    x2 = 0
    for v in range(v_number):
        x2 = x2 + result_df.iloc[1 + 5 * v, col - 1]
    result_df.iloc[row + 1, col - 1] = x2
    x3 = 0
    for v in range(v_number):
        x3 = x3 + result_df.iloc[2 + 5 * v, col - 1]
    result_df.iloc[row + 2, col - 1] = x3
    x4 = 0
    for v in range(v_number):
        x4 = x4 + result_df.iloc[3 + 5 * v, col - 1]
    result_df.iloc[row + 3, col - 1] = x4
    x5 = 0
    for v in range(v_number):
        x5 = x5 + result_df.iloc[4 + 5 * v, col - 1]
    result_df.iloc[row + 4, col - 1] = x5

    result_df2 = result_df.iloc[:, [0, -1]]
    result_df2 = result_df2.rename(columns={'SKU': '費用項目', '合計': '合計（円）'})
    result_df2 = result_df2.astype({'合計（円）': int})

    st.subheader(':mag:物流コスト')
    col3, col4 = st.columns((1, 1))
    with col4:
        st.dataframe(result_df2, width = 400, height = 800, hide_index = True)

    with col3:
        v_list = np.append(master_df0.Vendor.unique(), '全発注先').tolist()

        temp_df3 = result_df2.iloc[:result_df2.shape[0]-5, :]
        temp_df3['vendor'] = 'a'
        for i in range(len(v_list)-1):
            temp_df3.loc[0 + i * 5, 'vendor'] = v_list[i]
            temp_df3.loc[1 + i * 5, 'vendor'] = v_list[i]
            temp_df3.loc[2 + i * 5, 'vendor'] = v_list[i]
            temp_df3.loc[3 + i * 5, 'vendor'] = v_list[i]
            temp_df3.loc[4 + i * 5, 'vendor'] = v_list[i]
        temp_df3['cost'] = 'b'
        c_list = ['1)輸送コスト', '2)入庫コスト', '3)保管コスト', '4)欠品コスト', '5)総物流コスト']
        for i in range(len(c_list)):
            for j in range(4):
                temp_df3.loc[i + j * 5, 'cost'] = c_list[i]
        result_df3 = temp_df3.pivot_table(values='合計（円）', index='cost', 
                         columns='vendor')
        st.bar_chart(result_df3)

        v = st.radio(label=':orange[発注先を選択してください]',
                         options=v_list,
                         index=0,
                         horizontal=True,)
        temp_df2 = pd.DataFrame(np.append(master_df0.Vendor.unique(), '全発注先').tolist()).reset_index().set_index(0)
        v_num = temp_df2.loc[v, 'index']
        st.bar_chart(result_df2.iloc[v_num * 5 : v_num * 5 + 5, :], x = '費用項目', y = '合計（円）')

    st.subheader(':mag:各アイテムの在庫推移（m3）')
    col_list2 = ['SKU']
    for i in range(store_m3_all_df.shape[1]-1):
        col_list2.append(i)
    store_m3_all_df.columns = col_list2
    item_df = store_m3_all_df.reset_index()
    item_df = item_df[['SKU', 'index']].set_index('SKU')
    item = st.selectbox(label=':orange[アイテムを選択して下さい]',
                    options=item_df.index)
    item_num = item_df.loc[item,'index']
    st.line_chart(store_m3_all_df.loc[item_num, 1:])

