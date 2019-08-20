
# coding: utf-8

# ## Concept Image Enhance

# In[566]:

import os
import shutil
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as npimg

from tqdm import tqdm_notebook
from tqdm import tqdm


# In[567]:

# 이미지 Pretraining

par_folder = 'Dataset/22_AOI_HDI_dim1_Gvis'
data_paths = glob(par_folder+'/*/*.jpg')


# In[568]:

path = data_paths[3820]
img = npimg.imread(path)
print(path)


# In[570]:

img_rv = 256 - img

plt.figure(figsize = (10, 8))
plt.subplot(121)
plt.imshow(img)
plt.title('img_normal')
plt.subplot(122)
plt.imshow(img_rv)
plt.title('img_rv')
plt.show()


# In[573]:

cmaps = [plt.cm.copper, plt.cm.hot, 
         plt.cm.jet, plt.cm.hsv, plt.cm.RdBu, plt.cm.gist_rainbow ]
# plt.cm.gray, plt.cm.Greys, plt.cm.hot, plt.cm.afmhot, plt.cm.hot,

if not os.path.exists('test_image'):
        os.mkdir('test_image')

# savename = 'test_image/test_'+str(0)+'.jpg'
# plt.imsave(savename, img)

for i in range(len(cmaps)):
    savename = 'test_image/test_'+str(i)+'.jpg'
    plt.imsave(savename, img, cmap=cmaps[i])

for i in range(len(cmaps)):
    savename = 'test_image/test_'+str(i)+'_rv.jpg'
    plt.imsave(savename, img_rv, cmap=cmaps[i])


# In[577]:

par_folder1 = 'test_image'
data_paths1 = glob(par_folder1+'/*.jpg')
len(data_paths1)


# In[578]:

L = map(npimg.imread, data_paths1)
for i in range(len(data_paths1)):
    globals()['img{}'.format(i)] = next(L)


# In[580]:

imgs = [img0, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11]
titles = ['copper', 'copper_rv', 'hot', 'hot_rv', 'jet', 'jet_rv',
          'hsv', 'hsv_rv', 'RdBu', 'RdBu_rv', 'g_rainbow', 'g_rainbow_rv']
len(imgs)

num_image = len(imgs)
num_col = 6
a = num_image // num_col
plt.figure(figsize = (20, 8))

for i in range (num_image):
    #print(a, num_col, b+1)
    
    plt.subplot(a, num_col,i+1)
    plt.imshow(imgs[i][:,:,:])
    plt.title(titles[i])
plt.show()


# In[ ]:




# ## Enhanced Image Function

# In[ ]:




# In[1]:

import os
import shutil
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as npimg

from tqdm import tqdm_notebook
from tqdm import tqdm


def Enhance_image(x, y):
    par_folder = folder
    data_paths = glob(par_folder+'/*/*.jpg')
    
    # image conv plt.cm.type

    fold_tail = '_'+cmaps.name
    fold_tail_rv = fold_tail+'_rv'

    for cls_path in tqdm_notebook(os.listdir(par_folder)):
        dirname = os.path.join(par_folder, cls_path)
        data_paths_cls = glob(os.path.join(dirname, '*.jpg'))

        for path in data_paths_cls:
            new_dir = os.path.join(par_folder+fold_tail, cls_path)
            new_dir_rv = os.path.join(par_folder+fold_tail_rv, cls_path)

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            if not os.path.exists(new_dir_rv):
                os.makedirs(new_dir_rv)

            filename = os.path.basename(path).split('.')[0]+fold_tail+'.jpg'
            new_path = os.path.join(new_dir, filename)

            filename_rv = os.path.basename(path).split('.')[0]+fold_tail_rv+'.jpg'
            new_path_rv = os.path.join(new_dir_rv, filename_rv)

        # image conv
            img = npimg.imread(path)
            plt.imsave(new_path, img, cmap=cmaps)
            img_pil = Image.open(new_path)
            r1,g1,b1,__ = img_pil.split()
            img_conv = Image.merge('RGB',(r1,g1,b1))
            img_conv.save(new_path)

        # image conv_rv
            img_rv = 256-img
            plt.imsave(new_path_rv, img_rv, cmap=cmaps)
            img_pil_rv = Image.open(new_path_rv)
            r2,g2,b2,__ = img_pil_rv.split()
            img_rv_conv = Image.merge('RGB',(r2,g2,b2))
            img_rv_conv.save(new_path_rv)
           
    global par_folder_eh, par_folder_rv
    par_folder_eh = par_folder+fold_tail
    par_folder_rv = par_folder+fold_tail_rv
    print('par_folder_eh : ',par_folder_eh)
    print('par_folder_rv : ',par_folder_rv)


# In[2]:

def classfi_image(x):
    # Class 별 Test / Train으로 나누어 복사하기
    par_folder = x
    fold_tail = '_Cls'

    for cls_path in tqdm_notebook(os.listdir(par_folder)):
        dirname = os.path.join(par_folder, cls_path)
        data_paths = glob(os.path.join(dirname, '*.jpg'))

        for i, path in enumerate(data_paths):

    # 고정갯수로 나누기
            if i < 100:
                new_dir = os.path.join(par_folder+fold_tail,"test", cls_path)
            elif i >= 100 and i < 500:   # Max 수치 적용구분갯수 지정
                new_dir = os.path.join(par_folder+fold_tail,"train", cls_path)
            else:
                break

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            filename = os.path.basename(path)
            new_path = os.path.join(new_dir, filename)

            # rename files
            if not os.path.exists(new_path):
                os.rename(path, new_path)
            
#             # copy files
#             if not os.path.exists(new_path):
#                 shutil.copy2(path, new_path)

            # image resize 100,100
#             img_new = Image.open(new_path) 
#             img_re = img_new.resize((100,100))  
#             img_re.save(new_path)

    POR_path = par_folder + fold_tail + '/test/153_검은이물'
    NEW_path = par_folder + fold_tail + '/train/153_검은이물'

    data153_paths = glob(POR_path+'/*.jpg')
    len(data153_paths)

    if not os.path.exists(NEW_path):
        os.makedirs(NEW_path)

    for i, path in enumerate(data153_paths):
        if not i < 30:
            filename = os.path.basename(path)
            new_path = os.path.join(NEW_path, filename)
            os.rename(path, new_path)

    shutil.rmtree(par_folder)
    
    global par_folder_cls
    par_folder_cls=par_folder+fold_tail
    print('par_folder_cls : ',par_folder_cls)


# In[3]:

# train 이미지 불러서 90, 180, 270, 돌려서 뻥튀기 하고 저장하기 1 - 경로확인

def Augmt_image(x):

    from PIL import Image
    import matplotlib.pyplot as plt  # 새 창 안띄우고 그래프 실행
    import numpy as np

    dir_path1 = x
    dir_path2 = os.path.join(dir_path1,'train/*/','*.jpg')
    data_paths = glob(dir_path2)

    # 이미지 불러서 90, 180, 270, 돌려서 뻥튀기 하고 저장하기 2 - 돌리고 저장하기

    for path in tqdm_notebook(data_paths):
        image_pil = Image.open(path)

        image90 = image_pil.rotate(90)
        image180 = image_pil.rotate(180)
        image270 = image_pil.rotate(-90)

        save_path = os.path.dirname(path)        
        filename = os.path.basename(path)

        filename90 = os.path.splitext(filename)[0]+'_90.jpg'
        save_img90 = os.path.join(save_path,filename90)
        image90.save(save_img90)

        filename180 = os.path.splitext(filename)[0]+'_180.jpg'
        save_img180 = os.path.join(save_path,filename180)
        image180.save(save_img180)

        filename270 = os.path.splitext(filename)[0]+'_270.jpg'
        save_img270 = os.path.join(save_path,filename270)
        image270.save(save_img270)
    return print('Done.')


# In[ ]:




# ## Exec. Function

# In[14]:

# Enhance Image 1

# cmaps = [plt.cm.copper, plt.cm.hot, plt.cm.jet, plt.cm.hsv, plt.cm.RdBu, plt.cm.gist_rainbow ]

cmaps = plt.cm.gist_rainbow

folder = 'Dataset/22_AOI_HDI_dim1_Gvis'

Enhance_image(folder, cmaps)  # make image 2 types (cmap, revers cmap)


# In[15]:

# test / train  2-1

classfi_image(par_folder_eh)


# In[16]:

# image augmentation 2-1

Augmt_image(par_folder_cls)


# In[17]:

# test / train  2-2

classfi_image(par_folder_rv)


# In[18]:

# image augmentation 2-2

Augmt_image(par_folder_cls)


# In[ ]:




# In[25]:

# shutil.rmtree('Dataset/22_AOI_HDI_dim1_Gvis_copper_Cls')


# In[ ]:




# ## Class 별 Test / Train으로 나누어 복사하기

# In[ ]:

# Class 별 Test / Train으로 나누어 복사하기

par_folder = 'Dataset/22_AOI_HDI_dim1_Gvis'
fold_tail = '_Cls'

for cls_path in tqdm_notebook(os.listdir(par_folder)):
    dirname = os.path.join(par_folder, cls_path)
    data_paths = glob(os.path.join(dirname, '*.jpg'))
    
    for i, path in enumerate(data_paths):

# 고정갯수로 나누기
        if i < 100:
            new_dir = os.path.join(par_folder+fold_tail,"test", cls_path)
        elif i >= 100 and i < 500:   # Max 수치 적용구분갯수 지정
            new_dir = os.path.join(par_folder+fold_tail,"train", cls_path)
        else:
            break

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        
        filename = os.path.basename(path)
        new_path = os.path.join(new_dir, filename)
        
        if not os.path.exists(new_path):
            shutil.copy2(path, new_path)
            
        img_new = Image.open(new_path) 
        #img_re = img_new.resize((100,100))  # image resize 100,100
        img_re.save(new_path)

POR_path = par_folder + fold_tail + '/test/153_검은이물'
NEW_path = par_folder + fold_tail + '/train/153_검은이물'

data153_paths = glob(POR_path+'/*.jpg')
len(data153_paths)

if not os.path.exists(NEW_path):
    os.makedirs(NEW_path)

for i, path in enumerate(data153_paths):
    if not i < 30:
        filename = os.path.basename(path)
        new_path = os.path.join(NEW_path, filename)
        os.rename(path, new_path)


# ## 파일 절반 이동하기 / 삭제하기

# In[ ]:

# 파일 절반 이동하기
for i, fname in tqdm_notebook(enumerate(data_paths1)):
    if i % 2:
        new_path2 = fname.replace('Dataset/KSEM_HDI_AOI_Class/train/UNKNOWN','Dataset/KSEM_HDI_AOI_Class/train/NON_DEFECT')
        os.rename(fname,new_path2)


# In[ ]:

# 파일 절반 삭제하기 - 1 경로체크

path = 'Dataset/KSEM_HDI_AOI_Class3/train/DEFECT'
data_paths = glob(path+'/*.tif')
len(data_paths), data_paths[0]


# In[ ]:

# 파일 절반 삭제하기 - 2 실행
for i, fname in tqdm_notebook(enumerate(data_paths)):
    if i % 2:
        os.remove(fname)


# In[ ]:

for i, fname in tqdm_notebook(enumerate(data_paths1)):
    new_path2 = fname.replace('Dataset/KSEM_HDI_AOI_Class/train/UNKNOWN','Dataset/KSEM_HDI_AOI_Class/train/NON_DEFECT')
    os.rename(fname,new_path2)


# ## 조건부로 파일 이동하기

# In[ ]:

# 조건부로 파일 이동하기 1

FROM_PATH = 'Dataset/KSEM_HDI_AOI_Class/train/NON_DEFECT'
TO_PATH = 'Dataset/KSEM_HDI_AOI_Class/test/NON_DEFECT'

for i, fname in tqdm_notebook(enumerate(data_paths2)):
    if data_paths2[i].split('$')[9] == 'DEFECT' and count < 170:
        new_path = fname.replace(FROM_PATH,TO_PATH)
        os.rename(fname,new_path)
        


# In[ ]:

# 조건부로 파일 이동하기 2

FROM_PATH = 'Dataset/KSEM_HDI_AOI_Class/train/UNKNOWN'
TO_PATH = 'Dataset/KSEM_HDI_AOI_Class/test/UNKNOWN'

count = 0

for i, fname in tqdm_notebook(enumerate(data_paths2)):
    if data_paths2[i].split('$')[9] == 'DEFECT' and count < 385:
        new_path = fname.replace(FROM_PATH,TO_PATH)
        os.rename(fname,new_path)
        
        if data_paths2[i].split('$')[9] == 'DEFECT':
            count += 1
        else:
            pass


# In[ ]:




# ## 폴터 통채로 복사하기 (강제복사)

# In[ ]:

# 폴터 통채로 복사하기 (강제복사)
os.listdir('Dataset')


# In[ ]:

POR_path = 'Dataset/22_AOI_HDI_dim1_Class1'
NEW_path = 'Dataset/22_AOI_HDI_dim1_Class2'

shutil.copytree(POR_path, NEW_path)


# ## 조건부로 파일 삭제하기

# In[ ]:

# 조건부로 파일 삭제하기 - 경로
data_paths = glob('Dataset/KSEM_HDI_AOI_noCAD/*/*.tif')
len(data_paths),data_paths[0]


# In[ ]:

# 조건부로 파일 삭제하기 - 실행

for i, fname in tqdm_notebook(enumerate(data_paths)):
    if data_paths[i].split('$')[9] == 'CAM':
        os.remove(fname)


# ## 그냥 파일 이동하기

# In[ ]:

# 그냥 파일 이동하기 - 1 경로확인

FROM_PATH = 'Dataset/KSEM_HDI_AOI_noCAD/SHORT'
TO_PATH   = 'Dataset/KSEM_HDI_AOI_noCAD/Before'

data_paths = glob(FROM_PATH+'/*.tif')
len(data_paths), data_paths[0]


# In[ ]:

# 그냥 파일 다 이동하기 - 2 실행

for i, fname in tqdm_notebook(enumerate(data_paths)):
    new_path = fname.replace(FROM_PATH,TO_PATH)
    os.rename(fname,new_path)


# ## 폴더 지울 때 사용

# In[ ]:

# 통채로 삭제

import shutil
shutil.rmtree('Dataset/22_AOI_HDI_dim1_Gvis_copper_rv')


# In[ ]:

#폴더 지울 때 사용2
dir_path = par_folder
dir_list = os.listdir(dir_path)

for path in dir_list:
    if path[0] == '1' or path[0] == '2':
        del_path = os.path.join(par_folder, path)
        os.rmdir(del_path)


# ## 파일 pixel 자르고 따로 저장하기

# In[ ]:

# 파일 pixel 자르고 따로 저장하기 1 - 경로확인

data_paths = glob('Dataset/KSEM_HDI_AOI_origin2/Before/*.tif')
len(data_paths), data_paths[0]


# In[ ]:

# 파일 pixel 자르고 따로 저장하기 2 - 실행

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

for path in tqdm_notebook(data_paths):
    image_pil = Image.open(path)
    image_num = np.array(image_pil)
    cropImage = image_pil.crop((0, 0, 100, 100))
    
    filename = os.path.basename(path)
    filename2 = os.path.splitext(filename)[0]+'_resized.tif'
    save_path = 'Dataset/KSEM_HDI_AOI_origin2/Before2'
    save_img = os.path.join(save_path,filename2)
    cropImage.save(save_img)


# In[ ]:

# 구간별 실행

from PIL import Image
import matplotlib.pyplot as plt  # 새 창 안띄우고 그래프 실행
import numpy as np

path = data_paths[0]
image_pil = Image.open(path)
print('size :',image_pil.size)
image_pil


# In[ ]:

image_num = np.array(image_pil)   # image 를 숫자로 변환
image_num.shape


# In[ ]:

cropImage = image_pil.crop((0, 0, 100, 100)) # image 자르기 및 확인
print('crop size :',cropImage.size)
cropImage


# In[ ]:

# 파일 경로 설정 저장
save_path = 'Dataset/KSEM_HDI_AOI_origin2/Before2'
filename = os.path.basename(path)
filename2 = os.path.splitext(filename)[0]+'_resized.tif'
save_img = os.path.join(save_path,filename2)
save_img
cropImage.save(save_img)


# In[ ]:

cropImage.save(save_img) # 파일 저장


# ## Train image Augmentation

# In[ ]:

# train 이미지 불러서 90, 180, 270, 돌려서 뻥튀기 하고 저장하기 1 - 경로확인

from PIL import Image
import matplotlib.pyplot as plt  # 새 창 안띄우고 그래프 실행
import numpy as np

dir_path1 = par_folder + fold_tail
dir_path2 = os.path.join(dir_path1,'train/*','*.jpg')
data_paths = glob(dir_path2)
len(data_paths), data_paths[0]


# In[ ]:

# 이미지 불러서 90, 180, 270, 돌려서 뻥튀기 하고 저장하기 2 - 돌리고 저장하기

for path in tqdm_notebook(data_paths):
    image_pil = Image.open(path)
    
    image90 = image_pil.rotate(90)
    image180 = image_pil.rotate(180)
    image270 = image_pil.rotate(-90)
    
    save_path = os.path.dirname(path)        
    filename = os.path.basename(path)

    filename90 = os.path.splitext(filename)[0]+'_90.jpg'
    save_img90 = os.path.join(save_path,filename90)
    image90.save(save_img90)
    
    filename180 = os.path.splitext(filename)[0]+'_180.jpg'
    save_img180 = os.path.join(save_path,filename180)
    image180.save(save_img180)
    
    filename270 = os.path.splitext(filename)[0]+'_270.jpg'
    save_img270 = os.path.join(save_path,filename270)
    image270.save(save_img270)


# In[ ]:

# 이미지 3배 뻥튀기 : 불러서 90, 270, 돌려서 뻥튀기 하고 저장하기 2 - 돌리고 저장하기

for path in tqdm_notebook(data_paths):
    image_pil = Image.open(path)
    
    image90 = image_pil.rotate(90)
#    image180 = image_pil.rotate(180)
    image270 = image_pil.rotate(-90)
    
    save_path = os.path.dirname(path)        
    filename = os.path.basename(path)

    filename90 = os.path.splitext(filename)[0]+'_90.jpg'
    save_img90 = os.path.join(save_path,filename90)
    image90.save(save_img90)
    
#     filename180 = os.path.splitext(filename)[0]+'_180.jpg'
#     save_img180 = os.path.join(save_path,filename180)
#     image180.save(save_img180)
    
    filename270 = os.path.splitext(filename)[0]+'_270.jpg'
    save_img270 = os.path.join(save_path,filename270)
    image270.save(save_img270)


# In[ ]:



