# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
from numpyencoder import NumpyEncoder
import glob,os
from skimage import feature
import json
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt



path="C:/Users/Sritej. N/Desktop/cse515"
os.chdir(path)

# -----------------task0-----------------------------------------------------------
# The images_file below is a path for folder that has 400 images of olivetti dataset

images_file=path+"/faces"

def convertpng2array(path):
    im=Image.open(path).convert('L')
    i=np.array(im)
    i = i.astype('float32')
    i/=255
    return i

images_array=[]
'''
images_array stores the pixel values of all 400 images, images_array[2] gives us the array of image 0_2.png
images_array[13] gives us the array of image 1_3.png
'''
def createimagedata(path):
    index=0
    for idx in range(40):
        for subIdx in range(10):
            image_id=path+'/'+str(idx)+'_'+str(subIdx)+'.png'
            im=Image.open(image_id).convert('L')
            im=np.array(im)
            im = im.astype('float32')
            im/=255
            images_array.append(np.array(im))
            index+=1

createimagedata(images_file)

def mean(sub_img):
    sum=0
    for i in range(8):
        for j in range(8):
         sum+=sub_img[i][j]   
    return sum/64;

def standard_dev(sub_img,mean):
    sum=0
    for i in range(8):
        for j in range(8):
            sum+=(sub_img[i][j]-mean)**2
    return (sum/64)**(1./2.)

def skewness(sub_img,mean,std):
    sum=0.0
    for i in range(8):
        for j in range(8):
            sum+=(sub_img[i][j]-mean)**3
    
    res = sum/((sub_img.shape[0]*sub_img.shape[1]-1)*((std**3)))
    return res
    


def colormoments(img):
    colormoment = np.zeros(shape=(8, 8, 3))
    m = np.zeros(shape=(8, 8))
    stndard_d = np.zeros(shape=(8, 8))
    skwnes = np.zeros(shape=(8, 8))
    
    for i in range(8):
        for j in range(8):
            sub_img = img[i*8:(i+1)*8, j*8:(j+1)*8]
            m[i][j]=mean(sub_img)
            stndard_d[i][j]=standard_dev(sub_img,m[i][j])
            skwnes[i][j]=skewness(sub_img,m[i][j],stndard_d[i][j])
            colormoment[i][j][0]=m[i][j]
            colormoment[i][j][1]=stndard_d[i][j]
            colormoment[i][j][2]=skwnes[i][j]
    return  pd.DataFrame(colormoment.reshape(8*8,3)).to_numpy()

def Get_elbp(img):
    lbp = feature.local_binary_pattern(img, 8, 1, method='nri_uniform')
    results = []
    for i in range(8):
        for j in range(8):
            sub_img = lbp[i*8:(i+1)*8, j*8:(j+1)*8]
            (hist, _) = np.histogram(sub_img, density=True, bins=59, range=(0, 59))
            results.append(hist)
    return np.array(results).flatten().tolist()

def Get_hog(img):
    return feature.hog(img, orientations=9, 
        pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
            

'''
In task-1 with given an imageID ranging from 0 to 399 in Olivetti Dataset
& model which is either of cm,elbp or hog 
task1 function returns & prints that corresponding feature descriptor

'''
def task1(imgID,model):
    img=images_array[imgID]
    if(model.lower()=='cm'):
        cm=colormoments(img)
        print(cm)
        return cm
    elif(model.lower()=='elbp'):
        elbp=Get_elbp(img)
        print(elbp)
        return elbp
    elif(model.lower()=='hog'):
        hog=Get_hog(img)
        print(hog)
        return hog
    else:
        print("choose the correct model")
        return 0


'''
task2 function just takes path of the images folder 
& returns features of all three models for all the images in the folder 
features will be stored in a json file at given path 
-> code for storing json file is at the end of this file 
where task selection code is written

'''

def task2(path):
    features=[] 
    os.chdir(path)
    for file in glob.glob("*.png"):
        img_arr=convertpng2array(file)
        cm=colormoments(img_arr)
        elbp=Get_elbp(img_arr)
        hog=Get_hog(img_arr)
        feature={"id": file, "color_moment": cm,
                      "hog": hog, "elbp": elbp}
        features.append(feature)
    
    return features


def visualizeImg(path,targetImg,sortedSimilarity,k):
    plt.imshow(Image.open(path+'/'+targetImg).convert('L'))
    plt.show()
    c=0
    for Id,sim in sortedSimilarity:
        if(c==k):
            break
        c+=1
        plt.imshow(Image.open(path+'/'+Id).convert('L'))
        plt.show()
        print(c,' ',Id,':',sim)
    
def calculate_similarity(model,images_features,target_Img_features,ImgID):
    similarity={}
    feature=""
    
    if(model.lower()=='cm'):
        feature='color_moment'
        for img_f in images_features:
            sim=0
            if(img_f['id']!=ImgID):
                for i in range(len(img_f[feature])):
                    sim+=distance.cityblock(target_Img_features[feature][i],img_f[feature][i])
                
                similarity[img_f['id']]=np.mean(sim)
                
    if(model.lower()=='elbp' or model.lower()=='hog'):
        feature=model.lower()
        for img_f in images_features:
            if(img_f['id']!=ImgID):
                similarity[img_f['id']]=wasserstein_distance(target_Img_features[feature],img_f[feature])
                
    return similarity
'''
task3 function takes folder & Set as part of its parameteres 
which makes path to folder containing images 

ImgID is our target image with which we calculate similarity distance 
for all other images in the folder

it returns similarity in dict format w.r.t target image for each image in the folder for the given model

for ex:similarity['Image-5.png'] gives similarity distance of Image-5 from target_image
'''                
def task3(folder,Set,ImgID,model):
    path=folder+'/set'+str(Set)
    images_features=task2(path)
    target_Img_features={}
    similarity={}
    
    for image_features in images_features:
        if(image_features['id']==ImgID):
            target_Img_features=image_features            
    
    if(model.lower()=='cm' or model.lower()=='elbp' or model.lower()=='hog'):
        similarity=calculate_similarity(model, images_features, target_Img_features, ImgID)   
    
    if(model.lower()=='all'):
        cm_similarity=calculate_similarity('cm', images_features, target_Img_features, ImgID) 
        elbp_similarity=calculate_similarity('elbp', images_features, target_Img_features, ImgID) 
        hog_similarity=calculate_similarity('hog', images_features, target_Img_features, ImgID)
        for img_f in images_features:
            if(img_f['id']!=ImgID):
                ID=img_f['id']
                similarity[ID]=(1.0/25.0)*(cm_similarity[ID]) + elbp_similarity[ID] + (1.0/2.0)*(hog_similarity[ID])
        return {'simi':similarity,'cm':cm_similarity,'elbp':elbp_similarity,'hog':hog_similarity}
            
    return similarity


'''

Since task4 is similar to task3 and the only difference is 
we need to combine similarity from all 3 models
a if-case is added in the task3 itself where the distances from all 3 models 
are combined with a weightage to each model

'''

def task4(folder,Set,ImgID):
    return task3(folder,Set,ImgID,'all')
    

    
while(1):
    task=int(input("Select one of the options -> 1: task1, 2: task2, 3: task3, 4: task4, 5: exit \n"))
    if(task==1):   
         model=input("choose a model-> color momemts: cm, ELBP:elbp, HOG: hog \n")
         t1=task1(1,model)
         print(t1)

    elif(task==2):
        print("Enter path for folder containing images")
        images_file=input("for example -> 'C:/Users/Sritej. N/Desktop/cse515/faces' \n")
        feature_descriptors=task2(images_file)
        task2json=input("provide location for storing features, for ex:'C:/Users/Sritej. N/Desktop/cse515/task2.json'\n")
        with open(task2json, 'w') as outfile:
                    json.dump(feature_descriptors, outfile,cls=NumpyEncoder,indent=4)
        
    elif(task==3):
        folder=input("provide the folder location\n")
        #"C:/Users/Sritej. N/Desktop/cse515/test_imgage_sets"
        Set=input("choose the set\n")
        ImageID=input("provide imageID\n")
        model=input("select the model\n")
        k=int(input("enter the value of K or how many similar images needs to be visualized ?\n"))
        t3=task3(folder,Set,ImageID,model)
        t3=sorted(t3.items(), key=lambda x: x[1], reverse=False)
        
        visualizeImg(folder+'/set'+str(Set), ImageID, t3,k)

    elif(task==4):
        folder=input("provide the folder location\n")
        #"C:/Users/Sritej. N/Desktop/cse515/test_imgage_sets"
        Set=input("choose the set\n")
        ImageID=input("provide imageID\n")
        k=int(input("enter the value of K or how many similar images needs to be visualized ?\n"))
        t4=task4(folder,Set,'image-0.png')  
        t=sorted(t4['simi'].items(), key=lambda x: x[1], reverse=False)
        print('top4_image_Ids|overall_similarity| ColorMoment |  elbp  |   hog   ')
        for i in range(4):
            print(t[i][0],"        ","{:.2f}".format(t4['simi'][t[i][0]]),"         ","{:.2f}".format((1.0/25.0)*t4['cm'][t[i][0]]),"       ","{:.4f}".format(t4['elbp'][t[i][0]]),"    ","{:.3f}".format((1.0/2.0)*t4['hog'][t[i][0]]))
        
        visualizeImg(folder+'/set'+str(Set), ImageID, t,k)
        
    elif(task==5):
        break

    
    
    

