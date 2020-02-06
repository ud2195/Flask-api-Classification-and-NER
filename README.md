# Flask-api-Classification-and-NER

Proof of concept:

Description: Before a drug is released Pharmaceutical companies have to go through a process of submitting various regulatory documents in order to release the drug. In the process of doing the same our company already has a product where in a user has to do the following things:-

  1) Upload the document in a specific category(depending on the document by opening each and every file and checking the category it belongs to
  
  2) Open each and every document and check the drug substance it is talking about and upload the drug subtsnce name and the same process is to be followed for mentioning the Companies manufacturing the drug 
  
  3) If a document gets uploaded in a wrong category the consequences lead to no submission of document and time wastage 
  
  4) Imagine doing this for 10000 documents a day. Imagine the manual labour involved and the rate of error even if one document gets submitted in the wrong category or name of substance/substance manufacturer is wrong
  
  
  Solution(what i did):-
  
 • Extensive exploration of data led to me the following things
  1) The document can be scanned or a normal pdf(please check repository pdf to text)
  2) number of documents are 53(that is a huge number)
  3) Upon exploring the data i observed that a lot of documents had a consistent format as they were supposed to be submitted in a specific format which made me use regular expression to capture the data (NOT EVERYTHING REQUIRES MACHINE LEARNING)
  4) Rest of the documents were not standard across the industry so had to use a ML based approach
  5) Based upon the category predicted extract substance name , Manufacturer etc
  6) Build a flask api to perform above mentioned things
 
 
 • Not uploading the code for training NER models as i have already uploaded Substance-Ner   
  
