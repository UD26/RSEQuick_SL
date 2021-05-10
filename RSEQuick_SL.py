#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import time
import streamlit as st
import plotly.express as px
from datetime import datetime
import folium
from streamlit_folium import folium_static
import ee
import pandas as pd
from io import StringIO
import urllib, io, os
from skimage import filters
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import geehydro
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from PIL import Image
import base64

st.set_page_config(
    page_title="RSEQuick",
    page_icon="?????????????????????????????????????????????????????????",
    layout="wide",
)


def ee_initialize(token_name="EARTHENGINE_TOKEN"):
    """Authenticates Earth Engine and initialize an Earth Engine session"""
    if ee.data._credentials is None:
        ee_token = os.environ.get(token_name)
        if ee_token is not None:
            credential_file_path = os.path.expanduser("~/.config/earthengine/")
            if not os.path.exists(credential_file_path):
                credential = '{"refresh_token":"%s"}' % ee_token
                os.makedirs(credential_file_path, exist_ok=True)
                with open(credential_file_path + "credentials", "w") as file:
                    file.write(credential)
    ee.Initialize()

ee_initialize()



def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def AV (x):
    '''To interpolate Volume from Area '''
    for i in range (1,len(ARA)):
        x1=0
        x2=0
        y1=0
        y2=0
        if x>ARA[i]:
            pass
        else:
            x2=ARA[i]
            x1=ARA[i-1]
            y2=ARV[i]
            y1=ARV[i-1]
            break
    res=np.round(y1+((x-x1)*(y2-y1)/(x2-x1)),2)
    
#     print ("x:",x1,x2)
#     print ("y:",y1,y2)
    return res

def LV (x):
    '''To interpolate Volume from Level'''
    for i in range (1,len(ARL)):
        if x>ARL[i]:
            pass
        else:
            x2=ARL[i]
            x1=ARL[i-1]
            y2=ARV[i]
            y1=ARV[i-1]
            var = ((x-x1)*(y2-y1))/(x2-x1)
            res=np.round(y1+ var,2)
            
            break
#     print ("x:",x1,x2)
#     print ("y:",y1,y2)    
    return res


def LA (x):
    '''To interpolate area from Level'''
    for i in range (1,len(ARL)):
        if x>ARL[i]:
            pass
        else:
            x2=ARL[i]
            x1=ARL[i-1]
            y2=ARA[i]
            y1=ARA[i-1]
            var = ((x-x1)*(y2-y1))/(x2-x1)
            res=np.round(y1+ var,2)
            
            break
#     print ("x:",x1,x2)
#     print ("y:",y1,y2)
    return res

# trapezoidal formula
def simpson(A1,A2,H):
    '''trapezoidal formula'''
    return np.round(np.abs(H)*(A1+A2+np.sqrt(A1*A2))/3,2)


def clipping(image, geometry):
    return image.clip(geometry)


def estimateWSA(images,geometry,water_bands,water_threshold, water_sigma):
    geometry = ee.FeatureCollection(geometry)
    WSA_list = []
    global Dates_list
    Dates_list = []
    listOfImages = images.toList(images.size())

    Total_images = images.size().getInfo()
    
    global ndwi_list
    ndwi_list=[]
    for image_number in range(Total_images):

        image = ee.Image(listOfImages.get(image_number))
        ndwi = ee.Image(image).normalizedDifference(water_bands)
        
        ndwi_list.append(ndwi)
        
        edge = ee.Algorithms.CannyEdgeDetector(ndwi, water_threshold, water_sigma)
        ndwi_buffer = ndwi.mask(edge.focal_max(30, 'square', 'meters'))

        hist = ndwi_buffer.reduceRegion(ee.Reducer.histogram(150), geometry, 30).getInfo()
        values = ndwi_buffer.reduceRegion(ee.Reducer.toList(), geometry, 30).getInfo()

        th = filters.threshold_otsu(np.array(values['nd']))
        water = ndwi.gt(th)

        water_edge = ee.Algorithms.CannyEdgeDetector(water, 0.5, 0)

        area = ee.Image.pixelArea();
        waterArea = water.multiply(area).rename('waterArea');

        image = image.addBands(waterArea);

        stats = waterArea.reduceRegion(reducer=ee.Reducer.sum(),geometry= geometry,scale=30,crs = image.select('waterArea').projection())
        image_details = image.getInfo()

        WSA = stats.getInfo()

        Dates_list.append(image_details['properties']['DATE_ACQUIRED'])
        WSA_list.append(round((WSA['waterArea']/10**4),2))
        
    return Dates_list,WSA_list

water_sigma = 1
water_threshold = 0.5
water_bands = ['B3', 'B6']


def estimateRS(image_collection,geometry,start,stop,spillway_storage,path,row,cloud_cover,reservoir_data,water_level_data,sedimentation_duration,download,name=None):
                
        geometry = ee.FeatureCollection(geometry)

        global shapefile
        shapefile = geometry
        

        # generate percentile composite image
        images = ee.ImageCollection(image_collection)\
        .filterBounds(geometry)\
        .filter(ee.Filter.eq('WRS_PATH', path))\
        .filter(ee.Filter.eq('WRS_ROW', row))\
        .filterDate(start,stop).filterMetadata('CLOUD_COVER', 'less_than', cloud_cover)\
        #.map(clipping);

        dates,area = estimateWSA(images,geometry,water_bands,water_threshold,water_sigma)

        rs_area = pd.DataFrame(area,dates)
        rs_area.columns = ['waterArea']
        rs_area = rs_area/100 # convert to Mm2
        
        

        dam_df = reservoir_data
        observed_df = water_level_data

        
        global ARL, ARA, ARV
        ARL=dam_df['Level']
        ARA=dam_df['Area']
        ARV=dam_df['Capacity']
        
                       
        sort_df = {}
        for day in rs_area.index:
            if day in observed_df.index:
                sort_df[day]=np.float(observed_df.loc[day]['Level(m.)'])

        
        sort_df = pd.DataFrame(sort_df,index=['Level']).T
        sort_df['Area estimated'] = np.round(rs_area['waterArea'],2)


        # filtering wrong WSA estimation
        sorted_df1 = sort_df.sort_values(['Level','Area estimated'])
        sorted_df1['Original area']= sorted_df1['Level'].apply(LA)
        sorted_df1 = sorted_df1[sorted_df1['Area estimated']<sorted_df1['Original area']]
        
        sorted_df = pd.DataFrame(np.maximum.accumulate(sorted_df1['Area estimated']))
        sorted_df['Level'] = sorted_df1['Level']
        sorted_df = sorted_df.drop_duplicates(subset='Area estimated', keep="first")

        sorted_df['Original area']= sorted_df['Level'].apply(LA)
        
        #sorted_df['Original_capacity'] = sorted_df['Level'].apply(LV)- LV(sorted_df['Level'][0])
        
        estimated_capacity = np.zeros(len(sorted_df))
        original_capacity2 = np.zeros(len(sorted_df))
        for r in range(1,len(sorted_df)):
            A1 = sorted_df['Area estimated'][r-1]
            A2 = sorted_df['Area estimated'][r]
            H = sorted_df['Level'][r-1]-sorted_df['Level'][r]

            estimated_capacity[r]=simpson(A1,A2,H)

            Ao1 = sorted_df['Original area'][r-1]
            Ao2 = sorted_df['Original area'][r]
            original_capacity2[r]=simpson(Ao1,Ao2,H)
        
        sorted_df['Estimated capacity']= estimated_capacity
        sorted_df['Original capacity']= original_capacity2
        
        sorted_df['Cumulative estimated capacity']= sorted_df['Estimated capacity'].cumsum()
        sorted_df['Cumulative original capacity']= sorted_df['Original capacity'].cumsum()
        
        st.dataframe(sorted_df)
                
        
        #silt deposited b/w two consecutive surveys Mm3
        silt_deposited = np.round(sorted_df['Cumulative original capacity'][-1]-sorted_df['Cumulative estimated capacity'][-1],2)
        st.write("Reservoir capacity lost(MCM): "+str(silt_deposited))

        #rate of silt deposited b/w two consecutive surveys Mm3/year
        Periods_btw_surveys = sedimentation_duration
        rate_of_silt_deposited = np.round(silt_deposited/Periods_btw_surveys,2)
        st.write("Rate of siltation(MCM/yr): "+ str(rate_of_silt_deposited))

        # Life of reservoir
        total_capacity = spillway_storage # upto spillway crest
        life_of_reservoir = np.round(total_capacity/rate_of_silt_deposited,2)
        st.write("Life of reservoir(yrs): "+str(life_of_reservoir))


        #Revised Elevation-Capcity curves
        fig,ax = plt.subplots(figsize=(4,3))
        ax.plot(sorted_df['Level'],sorted_df[['Cumulative original capacity','Cumulative estimated capacity']])
        ax.legend(['Original capacity','Estimated capacity']);
        ax.set_xlabel('Level (m)')
        ax.set_ylabel('Capacity (MCM)')
        ax.set_title('Revised Elevation-Capacity curves');
        
        if name!=None:
            plt.savefig("Revised curves_"+str(name)+".jpg", dpi=300, bbox_inches='tight')
            st.markdown(get_binary_file_downloader_html("Revised curves_"+str(name)+".jpg", 'Download plot'), unsafe_allow_html=True)
            #To download file
            if download:
                sorted_df.to_csv('Sedimentation_analysis_'+str(name)+'.csv', header=True, index=True)
                st.markdown(get_binary_file_downloader_html('Sedimentation_analysis_'+str(name)+'.csv', 'Download data'), unsafe_allow_html=True)

        else:
            plt.savefig("Revised curves.jpg", dpi=300, bbox_inches='tight')
            st.markdown(get_binary_file_downloader_html("Revised curves.jpg", 'Download plot'), unsafe_allow_html=True)
            #To download file
            if download:
                sorted_df.to_csv('Sedimentation_analysis.csv', header=True, index=True)
                st.markdown(get_binary_file_downloader_html('Sedimentation_analysis.csv', 'Download data'), unsafe_allow_html=True)
        st.pyplot(fig)



########User Interface#####################


with st.form("my_form"):
    
    #st.title("RSEQuick")
    #st.markdown("Quick reservoir sedimentation assesment")
    
    image = Image.open('app_template.jpg')
    st.image(image)

    c1, c2, c3 = st.beta_columns((2, 1, 1))
    with c1:

        image_collections = st.selectbox(
            'Image collections',
            ("LANDSAT/LC08/C01/T1_TOA",
            "LANDSAT/LE07/C01/T1_RT_TOA",
            "LANDSAT/LT05/C01/T2_TOA",
            "LANDSAT/LT05/C01/T1_TOA"))



        geometry = st.text_input('Shapefile', 'Asset location')
       
        uploaded_reservoir_data = st.file_uploader(
            "Choose a Resevoir data file",type='csv')
        if uploaded_reservoir_data is not None:
            dataframe1 = pd.read_csv(uploaded_reservoir_data,sep=',',index_col=None)
            #st.write(dataframe1)

        uploaded_water_level = st.file_uploader(
            "Choose a Water level file",type='csv')
        if uploaded_water_level is not None:
            dataframe2 = pd.read_csv(uploaded_water_level,sep=',',index_col=0,parse_dates=True)
            #st.write(dataframe2)


    with c2:

        start_date = st.date_input(
            "Start",
            datetime(2021,1,1))
        start = str(start_date)

        path = st.number_input('Insert path number',format="%i",value=140)
        path = int(path)

        cloud_cover = st.slider(
            'Cloud cover % threshold', 0, 100, 5)

        spillway_storage = st.number_input('Active storage')


    with c3:

        end_date = st.date_input(
            "End",
            datetime(2021,1,1))
        end = str(end_date)

        row = st.number_input('Insert row number',format="%i",value=44)
        row = int(row)

        duration = st.slider(
            'Study duration', 0, 100, 6)
       
        name = st.text_input('Reservoir ID', 'Name')
        

        download = st.checkbox('Download files')

    submit_button = st.form_submit_button(label='Run')
    

    if submit_button:
        
        estimateRS(
            image_collections,
            geometry,
            start,
            end,
            spillway_storage,
            path,
            row,
            cloud_cover,
            dataframe1,
            dataframe2,
            duration,
            download,
            name)
        
        Map = folium.Map(location=[23.7175,85.8238], zoom_start=12,width="100%",height="100%")
        
        Map.setOptions('HYBRID')
        ndwiParams = {'min': -1, 'max': 1, 'palette': ['green', 'white', 'blue']}
        
        for num,img in enumerate(ndwi_list):
            Map.addLayer(img.clip(shapefile), ndwiParams, 'NDWI '+str(Dates_list[num]))
        
        Map.setControlVisibility(layerControl=True, fullscreenControl=True, latLngPopup=True)

        folium_static(Map)   
        
    else:
        st.write('Click here to start')


    


