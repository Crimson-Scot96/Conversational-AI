The trained model files can be located in the models folder on google drive:  
https://drive.google.com/drive/folders/1C_btnEOr36mlRzslE2X-PlFDlV9ElTpK?usp=drive_link  

### Quick start up guide:  

> If you want to **train the models yourself**:  

Step 1: Install the requirements file: "pip install -r requirements"  
Step 2:  
RASA: cd to this project and then use terminal and type "rasa train"  
BM25: Run bm25.py in the actions folder  
SBERT: Run finetuned_sbert.py in actions folder  

Step 3: in the terminal run: "rasa run actions" Once it says action point is running move on  
Step 4: Open a 2nd terminal and run: "rasa shell"  
Step 5: Ask your query and get a response  

> If you want to **use our pre trained models**:  

Step 1: Download the files via the google drive link above:  
* Rasa files: `models`   
* SBERT files: `actions -> output -> fine_tuned_sbert_model`    

Step 2: Install the requirements file: "pip install -r requirements"  
Step 3: in the terminal run: "rasa run actions" Once it says action point is running move on  
Step 4: Open a 2nd terminal and run: "rasa shell"  
Step 5: Ask your query and get a response  