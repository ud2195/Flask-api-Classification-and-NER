# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:16:31 2019

@author: udayk
"""
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess
import os


from flask import jsonify
from flask import Flask
from flask_restful import Resource, Api
from flask import request



app = Flask(__name__)



@app.route('/ectd',methods=['GET','POST'])
def dwac():
    #pdb.set_trace()
    req_data = request.get_json(force=True)

    Accountname = req_data['Accountname']
    #print(Accountname)
    Accountkey = req_data['Accountkey']
    #print(Accountkey)
    UserID= req_data['UserID']
    #print(UserID)
    FileID = req_data['FileID'][:]
    #print(FileID)
    block_blob_service = BlockBlobService(account_name=Accountname, account_key=Accountkey)
    #DIR='D:\\'
    #os.mkdir(DIR+UserID)
    #result={'UserID':UserID,'FileID': }
    user=[]
    fileid=[]
    result=[]
    d = defaultdict(list)
    for x in FileID:
        #print(x)
        block_blob_service.get_blob_to_path('documentcontainer',x,'D:\\azure api\\' +x+'.pdf')
        d['UserID'].append(UserID)
        #print(user)
        d['FileID'].append(x)
        #print(fileid)
        categories=convert('D:\\azure api\\' +x+'.pdf')
        #print(categories)
        d['Result'].append(categories)
        os.remove('D:\\azure api\\' +x+'.pdf')
        #print(user)
        #print(fileid)
        #print(result)
    d = [OrderedDict([('UserID', t), ('FileID', l),('Result',v)]) for t,l,v in zip(d['UserID'], d['FileID'],d['Result'])]    
    #y=list(zip(user,fileid,result))   
    return jsonify(d)
 
    #print (categories)
    
    #return jsonify(categories)
    
    
import requests    

x='murder+'

requests.get('https://indiankanoon.org/search/?formInput=)    
    
    
    9810609959

if __name__ == '__main__':
    app.run()
    
    

from collections import defaultdict
d = defaultdict(list)    
from collections import OrderedDict
    
    
    
    
    
    
    
    
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re
import io
from PIL import Image
import pytesseract
from wand.image import Image as wi
from sklearn.externals import joblib
import csv
import json
   
    import pdb
    
    
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow    
    
    
import numpy as np
    

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import seaborn as sns
import re
from nltk.tokenize import RegexpTokenizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from nltk.stem import WordNetLemmatizer,PorterStemmer    
    
    
    
    
def convert(fname, pages=None,encoding='utf-8'):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    print(text)
    #pdb.set_trace()
    #pdb.set_trace()
    if len(text)>=500:
        regex3=re.search(r"[3.]\d*(?:[.-]\w+)*\s*(Nomenclature|General Information|Process validation|Justification of Specification(s)|Manufacturer(s)|Batch Formula|Description of Manufacturing Process and Process Controls|Controls of Critical Steps and Intermediates|Process Validation and or Evaluation|Specification(s)|Analytical Procedures|Validation of Analytical Procedures|Batch Analyses|Characterization of Impurities|Reference Standards or Materials|Container Closure System|Pharmaceutical Development|Description and Composition of the Drug Product|Quality overall summary|Nomenclature|Structure|General properties|Manufacturer|Description of Manufacturing Process and Process Controls|Stability Data|Control of Materials|Controls of Critical Steps and Intermediates|Process Validation and/or Evaluation|Manufacturing Process Development|Characterization|Elucidation of Structure and other Characteristics|Impurities|Specification|Analytical Procedures|Validation of Analytical Procedures|Batch Analyses|Justification of Specification|Reference Standards or Materials|Container Closure Systems|Stability Summary and Conclusions|Post Approval Stability Protocol and Stability Commitment)",text,re.IGNORECASE)
        #print(regex3.group())
        if regex3 is not None and str(regex3.group())[:5]==('3.2.S' or '3.2.s'):
            
            substancemanufacturer = []
            od={}
            with open(r'C:\Users\udayk\Downloads\organizations.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            od['Substance Manufacturer']=a
                            break
            od['Substance Manufacturer']=''            
            od['Dosage'] = ''
            #with open(r'C:\Users\udayk\Downloads\dosage.csv', newline='', encoding ='latin1') as myFile:
            #    reader = csv.reader(myFile)
            #    for row in reader:
            #        if len(row[1])>=4:
            #            w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
            #            if w: 
            #                b = w.group()
            #                dosage.append(b)
            #                break
            substances = [] 
            with open(r'C:\Users\udayk\Downloads\substances.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if z: 
                            c=z.group()
                            od['Substance']=c
                            break
            od['Substance']=''                
            od['Product Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
            od['Category']=regex3.group()
            return od#regex3.group(),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
            
        elif regex3 is not None and str(regex3.group())[:5]==('3.2.P' or '3.2.p'):
            
            
            productmanufacturer = []
            od={}
            with open(r'C:\Users\udayk\Downloads\organizations.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            productmanufacturer.append(a)
                            od['Product Manufacturer']=a
                            break
            od['Product Manufacturer']=''            
            dosage = []
            with open(r'C:\Users\udayk\Downloads\dosage.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if w: 
                            b = w.group(0)
                            dosage.append(b)
                            od['Dosage']=b
                            break
            od['Dosage']=''            
            od['Substance'] = '' 
            #with open(r'C:\Users\udayk\Downloads\substances.csv', newline='', encoding ='latin1') as myFile:
            #    reader = csv.reader(myFile)
            #    for row in reader:
            #        if len(row[1])>=4:
            #            z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
            #            if z: 
            #                c=z.group(0)
            #                substances.append(c)
            #               break                            
            od['Substance Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
            od['Category']=regex3.group()
            return od#regex3.group(),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
            
        #elif regex3:
         #   productmanufacturer = []
          #  with open(r'C:\Users\udayk\Downloads\organizations.csv', newline='', encoding ='latin1') as myFile:
           #     reader = csv.reader(myFile)
            #    for row in reader:
             #       if len(row[1])>=4:
              #          v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
               #         if v: 
                #            a = v.group()
                 #           productmanufacturer.append(a)
                  #          break
            #dosage = []
            #with open(r'C:\Users\udayk\Downloads\dosage.csv', newline='', encoding ='latin1') as myFile:
             #   reader = csv.reader(myFile)
             #   for row in reader:
              #      if len(row[1])>=4:
               #         w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                #        if w: 
                 #           b = w.group()
                  #          dosage.append(b)
                   #         break
            #substances = [] 
            #with open(r'C:\Users\udayk\Downloads\substances.csv', newline='', encoding ='latin1') as myFile:
             #   reader = csv.reader(myFile)
              #  for row in reader:
               #     if len(row[1])>=4:
                #        z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                 #       if z: 
                  #          c=z.group()
                   #         substances.append(c)
                    #        break                            

            #substancemanufacturer=[]
            #probabilityscore=1.00
            #product=' '
            #return regex3.group(),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
        
        elif regex3 is None:
            od={}
            def preprocess(sentence):
                sentence=str(sentence)
                sentence = sentence.lower()
                sentence=sentence.replace('{html}',"") 
                cleanr = re.compile('<.*?>')
                cleantext = re.sub(cleanr, '', sentence)
                rem_url=re.sub(r'http\S+', '',cleantext)
                rem_num = re.sub('[0-9]+', '', rem_url)
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(rem_num)  
                filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
                #stem_words=[stemmer.stem(w) for w in filtered_words]
                lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
                return " ".join(lemma_words)
            
            preprocess(text)
            sgd_model = joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\smotesvm.pkl')
            count_vect=joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\count_vect.pickel')
            tfidf_transformer=joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\tfidf_transformer.pickel')
            #count_vect = CountVectorizer()
            X_train_counts = count_vect.transform([text])
            #tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.transform(X_train_counts)
            print(X_train_tfidf.dtype)
            #X_train_tfidf=np.array(list(X_train_tfidf), dtype=np.float)
            print(X_train_tfidf)

            prediction=sgd_model.predict(X_train_tfidf)
            #ynew = sgd_model.predict_proba([text])
            #for z in ynew:
            #    probabilityscore=max(z)
            od['Product Manufacturer']=''
            od['Substance Manufacturer']=''
            od['Dosage']=''
            od['Substance']=''
            od['Product']=''
            od['Category']=str(prediction)
            #od['Probability score']=str(probabilityscore)
            return od#str(prediction),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
#            return ''.join(map(str, prediction))
        else:
            print("Search returned zero values")
            
                                                            
    else:
        pdffile = wi(filename = fname, resolution = 300)
        pdfImg = pdffile.convert('jpeg')

        imgBlobs = []

        for img in pdfImg.sequence:
            page = wi(image = img)
            imgBlobs.append(page.make_blob('jpeg'))
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"    
        for imgBlob in imgBlobs:
            im= Image.open(io.BytesIO(imgBlob))
            text2 = pytesseract.image_to_string(im, lang = 'eng')
        regex3=re.search(r"[3.]\d*(?:[.-]\w+)*\s*(Nomenclature|General Information|Process validation|Justification of Specification(s)|Manufacturer(s)|Batch Formula|Description of Manufacturing Process and Process Controls|Controls of Critical Steps and Intermediates|Process Validation and or Evaluation|Specification(s)|Analytical Procedures|Validation of Analytical Procedures|Batch Analyses|Characterization of Impurities|Reference Standards or Materials|Container Closure System|Pharmaceutical Development|Description and Composition of the Drug Product|Quality overall summary|Nomenclature|Structure|General properties|Manufacturer|Description of Manufacturing Process and Process Controls|Stability Data|Control of Materials|Controls of Critical Steps and Intermediates|Process Validation and/or Evaluation|Manufacturing Process Development|Characterization|Elucidation of Structure and other Characteristics|Impurities|Specification|Analytical Procedures|Validation of Analytical Procedures|Batch Analyses|Justification of Specification|Reference Standards or Materials|Container Closure Systems|Stability Summary and Conclusions|Post Approval Stability Protocol and Stability Commitment)",text2,re.IGNORECASE)
        print(regex3)
        if regex3 is not None and str(regex3.group())[:5]==('3.2.S' or '3.2.s'):
            substancemanufacturer = []
            od={}
            with open(r'C:\Users\udayk\Downloads\organizations.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text2, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            od['Substance Manufacturer']=a
                            break
            od['Substance Manufacturer']=''            
            od['Dosage'] = ''
            #with open(r'C:\Users\udayk\Downloads\dosage.csv', newline='', encoding ='latin1') as myFile:
            #    reader = csv.reader(myFile)
            #    for row in reader:
            #        if len(row[1])>=4:
            #            w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
            #            if w: 
            #                b = w.group()
            #                dosage.append(b)
            #                break
            substances = [] 
            with open(r'C:\Users\udayk\Downloads\substances.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if z: 
                            c=z.group()
                            substances.append(c)
                            od['Substance']=c
                            break
            od['Substance']=''                
            od['Product Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
            od['Category']=regex3.group()
            return od#regex3.group(),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
            
        elif regex3 is not None and str(regex3.group())[:5]==('3.2.P' or '3.2.p'):
            
            
            #if regex3.group()=='3.2.P.3.1 Manufacturer(s)':
            productmanufacturer = []
            od={}
            with open(r'C:\Users\udayk\Downloads\organizations.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            od['Product Manufacturer']=a
                            break
            od['Product Manufacturer']=''            
            dosage = []
            with open(r'C:\Users\udayk\Downloads\dosage.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if w: 
                            b = w.group()
                            od['Dosage']=b
                            
                            break
            od['Dosage']=''            
            od['Substance'] = '' 
            #with open(r'C:\Users\udayk\Downloads\substances.csv', newline='', encoding ='latin1') as myFile:
            #    reader = csv.reader(myFile)
            #    for row in reader:
            #        if len(row[1])>=4:
            #           z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
            #            if z: 
            #                c=z.group()
            #                substances.append(c)
            #                break                            
            od['Substance Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
            od['Category']=regex3.group()
            return od#regex3.group(),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
            
        #elif regex3:
         #   productmanufacturer = []
          #  with open(r'C:\Users\udayk\Downloads\organizations.csv', newline='', encoding ='latin1') as myFile:
           #     reader = csv.reader(myFile)
            #    for row in reader:
             #       if len(row[1])>=4:
              #          v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
               #         if v: 
                #            a = v.group()
                 #           productmanufacturer.append(a)
                  #          break
            #dosage = []
            #with open(r'C:\Users\udayk\Downloads\dosage.csv', newline='', encoding ='latin1') as myFile:
             #   reader = csv.reader(myFile)
              #  for row in reader:
               #     if len(row[1])>=4:
                #        w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                 #       if w: 
                  #          b = w.group()
                   #         dosage.append(b)
                    #        break
            #substances = [] 
            #with open(r'C:\Users\udayk\Downloads\substances.csv', newline='', encoding ='latin1') as myFile:
             #   reader = csv.reader(myFile)
              #  for row in reader:
               #     if len(row[1])>=4:
                #        z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                 #       if z: 
                  #          c=z.group()
                   #         substances.append(c)
                    #        break                            

            #substancemanufacturer=[]
            #probabilityscore=1.00
            #product=' '
            #return regex3.group(),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
        
        elif regex3 is None:
            od={}
            def preprocess(sentence):
                sentence=str(sentence)
                sentence = sentence.lower()
                sentence=sentence.replace('{html}',"") 
                cleanr = re.compile('<.*?>')
                cleantext = re.sub(cleanr, '', sentence)
                rem_url=re.sub(r'http\S+', '',cleantext)
                rem_num = re.sub('[0-9]+', '', rem_url)
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(rem_num)  
                filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
                #stem_words=[stemmer.stem(w) for w in filtered_words]
                lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
                return " ".join(lemma_words)
            
            preprocess(text2)
            sgd_model = joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\smotesvm.pkl')
            count_vect=joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\count_vect.pickel')
            tfidf_transformer=joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\tfidf_transformer.pickel')
            #count_vect = CountVectorizer()
            X_train_counts = count_vect.transform([text])
            #tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.transform(X_train_counts)
            print(X_train_tfidf.dtype)
            #X_train_tfidf=np.array(list(X_train_tfidf), dtype=np.float)
            print(X_train_tfidf)

            prediction=sgd_model.predict(X_train_tfidf)
            
            #prediction=sgd_model.predict([text])
            #ynew = sgd_model.predict_proba([text])
            #for z in ynew:
            #    probabilityscore=max(z)
            od['Product Manufacturer']=''
            od['Substance Manufacturer']=''
            od['Dosage']=''
            od['Substance']=''
            od['Product']=''
            od['Category']=str(prediction)
            #od['Probability score']=str(probabilityscore)
            return od#str(prediction),str(probabilityscore),product,productmanufacturer,dosage,substances,substancemanufacturer
#            return ''.join(map(str, prediction))
        else:
            print("Search returned zero values")
    



convert(r'C:\Users\udayk\Documents\My Received Files\fi-cover-1.pdf')
count_vect=joblib.load(r'D:\DownloadFiles (2)\DownloadFiles\text files\count_vect.pickel')
convert(r'C:\Users\udayk\Documents\My Received Files\1.4.1 LOA_d3666eed-05d5-4e90-950a-102dff3fd013.pdf')


import tensorflow_hub as hub
import tensorflow as tf






from timeit import timeit
import re

def find(string, text):
    if string.index(text):
        pass

def re_find(string, text):
    if re.match(text, string):
        pass

def best_find(string, text):
    if text in string:
       pass

print (timeit("find(string, text)", "from __main__ import find; string='lookforme'; text='look'"))  
print (timeit("re_find(string, text)", "from __main__ import re_find; string='lookforme'; text='look'"))  
print (timeit("best_find(string, text)", "from __main__ import best_find; string='look for me'; text='look'"))





