import sys
import os
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import cv2
from collections import Counter
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QProgressBar

def get_domain_color(image, k=10, n=3):
    if k < n:
        raise Exception('Amount of clusters cant be more than number of dominant color.')
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k, algorithm='auto')
    labels = clt.fit_predict(image)
    label_counter = Counter(labels)
    dominant_clrs = []
    for i in np.arange(0, n):
        dominant_clrs.append(clt.cluster_centers_[label_counter.most_common(n)[i][0]])
    return dominant_clrs

def analyse_img(name, way):
    data = []
    fn = name
    img = cv2.imread(fn)
    img = cv2.resize(img, (480,360))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Dominant colors
    n_clusters = 8
    dom_col = get_domain_color(img, k = n_clusters, n = n_clusters)[0:5]
    dom_clr = np.zeros([img.shape[0], 1,3], dtype = 'uint8')
    for i in np.arange(0,5):
        data.append(dom_col[i][0])
        data.append(dom_col[i][1])
        data.append(dom_col[i][2])
        dom_clr = np.hstack((dom_clr,np.full((img.shape[0], img.shape[1]//5,3), dom_col[i], dtype = 'uint8')))
    out_img = np.hstack((img, dom_clr))
    cv2.imwrite(way + './dom_clr/'+name.split('/')[len(name.split('/'))-1],out_img)
    # Picture contours filling
    canny = cv2.Canny(imgray, 120, 50)
    filling = np.zeros((3,3))
    for line in np.arange(0,3):
        for column in np.arange(0,3):
            data.append(canny[120*column:120*(column+1)-1,160*line:160*(line+1)-1].mean()/255)
    #corners count
    thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,3)
    thresh_smoth = cv2.GaussianBlur(thresh,(7,7),1)
    dst = cv2.cornerHarris(thresh_smoth, 9,3, 0.2)
    dst = cv2.cornerHarris(thresh_smoth, 2, 3, 0.04)
    dst = cv2.dilate(dst,None)
    data.append(img[dst>0.1*dst.max()].shape[0])
    #count elipses
    thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,3)
    contours0, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elipse_count = 0
    for cnt in contours0:
        if len(cnt)>10 and len(cnt)<480:
            elipse_count +=1
    data.append(elipse_count)
    return np.array(data)

class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()

        self.way = ""
        self.completed = 0
        self.setGeometry(250, 250, 300, 110)
        self.setFixedSize(300, 110)
        self.setWindowTitle("Analysis in progress")
        self.button = QPushButton('Start', self)
        self.button.move(110, 58)
        self.button.clicked.connect(self.onButtonClick)
        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setGeometry(30, 30, 240, 20)

        self.show()

    def onButtonClick(self):
        if self.button.text() == "Stop" or self.button.text() == "Done!":
            quit()
        if self.button.text() == "Start":
            string = str(QFileDialog.getExistingDirectory())
            QFileDialog().close()
            if len(string) == 0:
                return
            self.button.setText("Stop")
            #prepare folder for pics
            try:
                os.mkdir(string + '/dom_clr/')
            except FileExistsError as exep:
                print("dom_clr folder is alredy exist")
            #start analysis
            self.start(string)


    def start(self, way):

        self.way = way
        img_list = glob.glob(way + '/*.*')
        #step
        step = 99/len(img_list)
        #bad code below
        dataset = analyse_img(img_list[0], self.way)
        for img in img_list:
            dataset = np.vstack([dataset, analyse_img(img, self.way)])
            self.completed += step
            self.progress.setValue(self.completed)
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename({
            0: 'dom_col_1_B',
            1: 'dom_col_1_G',
            2: 'dom_col_1_R',
            3: 'dom_col_2_B',
            4: 'dom_col_2_G',
            5: 'dom_col_2_R',
            6: 'dom_col_3_B',
            7: 'dom_col_3_G',
            8: 'dom_col_3_R',
            9: 'dom_col_4_B',
            10: 'dom_col_4_G',
            11: 'dom_col_4_R',
            12: 'dom_col_5_B',
            13: 'dom_col_5_G',
            14: 'dom_col_5_R',
            15: '(0,0) fil',
            16: '(0,1) fil',
            17: '(0,2) fil',
            18: '(1,0) fil',
            19: '(1,1) fil',
            20: '(1,2) fil',
            21: '(2,0) fil',
            22: '(2,1) fil',
            23: '(2,2) fil',
            24: 'corners_cnt',
            25: 'elipse_cnt'}, axis=1)
        dataset.to_excel(way + '/dom_clr/meta_data.xlsx')

        self.progress.setValue(100)
        self.button.setText("Done!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
