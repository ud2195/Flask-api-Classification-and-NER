from fuzzywuzzy import fuzz
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
import spacy


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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from nltk.stem import WordNetLemmatizer,PorterStemmer    


import pdb
   
        
    
def convert(fname, pages=None,encoding='utf-8'):
    #pdb.set_trace()
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)
    #pdb.set_trace()

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
    #print(text)
    if len(text)>=500:
        #od['Category']=''
        regex3=re.search(r"[3.]\d*(?:[.-]\w+)*\s*(#cannot mention the categories)",text,re.IGNORECASE)
        if regex3 is not None:
            mn={}
            tobematched=[#cannot mention the categories]
            od={}
            for x in tobematched:
                
                Ratio=fuzz.ratio(regex3.group().lower(),x.lower())
                mn[x]=Ratio
            regex3=max(mn,key=mn.get)    
            od['Substance Manufacturer']=''            
            od['Dosage'] = ''
            od['Substance']=''                            
            od['Product Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
            od['Category']=regex3
            substancemanufacturer = []
            with open('xyz', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            od['Substance Manufacturer']=a
                            substancemanufacturer.append(a)
                            break
                        
            if substancemanufacturer is None:
                nlp = spacy.load('xyz/org_spacy')
                doc = nlp(text)
                for ent in doc.ents:
                    new_ents = ent.text
                    od['Substance Manufacturer']= new_ents
        
            substances = []
            
            with open('/home/daniel/Arindam/Python_scripts/emea_substance.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if z: 
                            c=z.group()
                            od['Substance']=c
                            substances.append(c)
                            break
                        
            if substances is None:        
                nlp = spacy.load('xyz/sub_spacy')
                doc = nlp(text)
                for ent in doc.ents:
                    new_ents = ent.text
                    od['Substance']= new_ents
                    
            
            #return od                                    
            productmanufacturer = []
            
            with open('xyz/emea_org.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            productmanufacturer.append(a)
                            od['Product Manufacturer']=a
                            break
                        
            if productmanufacturer is None :
                nlp = spacy.load('xyz/org_spacy')
                doc = nlp(text)
                for ent in doc.ents:
                    new_ents = ent.text
                    od['Product Manufacturer']= new_ents
            
            
            with open('xyz/dosage.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if w: 
                            b = w.group(0)
                            #dosage.append(b)
                            od['Dosage']=b
                            break
            return od
        
        if regex3 is None:
            
            od={}
            od['Substance Manufacturer']=''            
            od['Dosage'] = ''
            od['Substance']=''                            
            od['Product Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
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
                
                lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
                return " ".join(lemma_words)


            text=preprocess(text)
            #pdb.set_trace()
            model = joblib.load(r'D:\xyz\gsmlp.pkl')
            count_vect=joblib.load(r'D:\xyz\count_vect.pickel')
            tfidf_transformer=joblib.load(r'D:\xyz\tfidf_transformer.pickel')
            labelencoder=joblib.load(r'D:\xyz\number.pickel')
            text = count_vect.transform([text])
            text = tfidf_transformer.transform(text)
            prediction=model.predict(text)
            prediction=labelencoder.inverse_transform([prediction])
            
            od['Category']=str(prediction)
            
            return od
        else:
            return "Search returned zero values"
                                                            
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
        regex3=re.search(r"[3.]\d*(?:[.-]\w+)*\s*(#cannot mention categories)",text2,re.IGNORECASE)
        if regex3 is not None:
            mn={}
            tobematched=['#cannot mention categories']
            od={}
            for x in tobematched:
                Ratio=fuzz.ratio(regex3.group().lower(),x.lower())
                mn[x]=Ratio
            regex3=max(mn,key=mn.get)    
            od={}
            od['Substance Manufacturer']=''            
            od['Dosage'] = ''
            od['Substance']=''                            
            od['Product Manufacturer']=''
            od['Probability score']=1.00
            od['Product']=''
            od['Category']=regex3
            substancemanufacturer = []
            with open('xyz/emea_org.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            od['Substance Manufacturer']=a
                            substancemanufacturer.append(a)
                            break
            if substancemanufacturer is None:
                nlp = spacy.load('/home/daniel/Arindam/Python_scripts/org_spacy')
                doc = nlp(text)
                for ent in doc.ents:
                    new_ents = ent.text
                    od['Substance Manufacturer']= new_ents
            substances = [] 
            with open('xyz/emea_substance.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        z = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if z: 
                            c=z.group()
                            substances.append(c)
                            od['Substance']=c
                            break
            if substances is None:        
                nlp = spacy.load('xyz/sub_spacy')
                doc = nlp(text)
                for ent in doc.ents:
                    new_ents = ent.text
                    od['Substance']= new_ents
            od['Category']=regex3.group()
            #return od
                                    
            productmanufacturer = []
            with open('xyz/emea_org.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        v = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if v: 
                            a = v.group()
                            od['Product Manufacturer']=a
                            productmanufacturer.append(a)
                            break
            if productmanufacturer is None:            
                nlp = spacy.load('/home/daniel/Arindam/Python_scripts/org_spacy')
                doc = nlp(text)
                for ent in doc.ents:
                    new_ents = ent.text
                    od['Product Manufacturer']= new_ents
            #dosage = []
            with open('xyz/dosage.csv', newline='', encoding ='latin1') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    if len(row[1])>=4:
                        w = re.search(r'\b' + re.escape(row[1]) + r'\b', text, re.IGNORECASE)
                        if w: 
                            b = w.group()
                            od['Dosage']=b                    
                            break
            return od

    if regex3 is None:
        #print('THERE')
        od={}
        od['Substance Manufacturer']=''            
        od['Dosage'] = ''
        od['Substance']=''                            
        od['Product Manufacturer']=''
        od['Probability score']=1.00
        od['Product']=''
        #od['Category']=regex3.group()
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


        text=preprocess(text2)
        model = joblib.load(r'D:\work for auto ectd\gsmlp.pkl')
        count_vect=joblib.load(r'D:\work for auto ectd\count_vect.pickel')
        tfidf_transformer=joblib.load(r'D:\work for auto ectd\tfidf_transformer.pickel')
        labelencoder=joblib.load(r'D:\work for auto ectd\number.pickel')
        text = count_vect.transform([text])
        text = tfidf_transformer.transform(text)
        prediction=model.predict(text)
        prediction=labelencoder.inverse_transform([prediction])

        
        
        
        
        
        
        
        #for z in ynew:
        #    probabilityscore=max(z)
        od['Category']=str(prediction)
        #od['Probability score']=str(probabilityscore)
        return od
    else:
        return "Search returned zero values"
    
    
    
convert(r'xyz\1131summaryfornonclinicalstudy_9a79b2c1-d537-437f-bc1a-5487dc0529c6.pdf')    
    
    
    
    
    
    
    
    
    
    
    
