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
    
    Accountkey = req_data['Accountkey']
    
    UserID= req_data['UserID']
    
    FileID = req_data['FileID'][:]
    
    block_blob_service = BlockBlobService(account_name=Accountname, account_key=Accountkey)
    
    user=[]
    fileid=[]
    result=[]
    d = defaultdict(list)
    for x in FileID:
        #print(x)
        block_blob_service.get_blob_to_path('documentcontainer',x,'D:\\azure api\\' +x+'.pdf')
        d['UserID'].append(UserID)
       
        d['FileID'].append(x)
        #print(fileid)
        categories=convert('D:\\azure api\\' +x+'.pdf')(#refer to prediction and extraction)
        
        d['Result'].append(categories)
        os.remove('D:\\azure api\\' +x+'.pdf')
 
       
    d = [OrderedDict([('UserID', t), ('FileID', l),('Result',v)]) for t,l,v in zip(d['UserID'], d['FileID'],d['Result'])]    
      
    return jsonify(d)


if __name__ == '__main__':
    app.run()

 
    
